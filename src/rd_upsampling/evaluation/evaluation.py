import tensorflow as tf
import gin
import logging
from tqdm import tqdm
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import os

from evaluation.evaluation_loss_functions import loss_function, sdr, lsd, generator_loss, discriminator_loss, weighted_mse, vgg_perceptual_loss
from evaluation.visualization import resolution_calculation, synthesis_13_rd, synthesis_12_super_rd, synthesis_13_super_rd, synthesis_33_rd, synthesis_22_super_rd, synthesis_22_rd, synthesis_23_super_rd, synthesis_23_rd
from utils.data_processing import normalisation_processing, normalisation_back

from models.dp_transformer_stft_exponential import DPTransformerSTFT
from models.dp_test import DPTransformerSTFT_test
from models.swinir_tensorflow import SwinIR
from models.cGAN import ConditionalGAN
from models.architectures import interpolation_upsampling, CNN_simple, UNet_simple, UNet_concat


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# general evaluation function
@gin.configurable
class Evaluation():
    def __init__(self, ds_test, model, benchmark_model, checkpoint_restore_path, benchmark_checkpoint_path, ckpt_max_to_keep, processing_method, layer_names, whether_fftshift):
        self.ds_test = ds_test
        self.model = model
        self.benchmark_model = benchmark_model
        self.checkpoint_restore_path = checkpoint_restore_path
        self.benchmark_checkpoint_path = benchmark_checkpoint_path
        self.ckpt_max_to_keep = ckpt_max_to_keep
        self.processing_method = processing_method
        self.layer_names = layer_names
        self.whether_fftshift = whether_fftshift

        self.eval_step_loss = tf.keras.metrics.Mean()
        self.eval_benchmark_loss = tf.keras.metrics.Mean()

        self.range_res, self.velocity_res = resolution_calculation()

        logging.info("------------------------Start the evaluation process------------------------")

        # Restore the latest checkpoint in checkpoint_dir
        self.ckpt = tf.train.Checkpoint(net=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, directory=self.checkpoint_restore_path, max_to_keep=self.ckpt_max_to_keep)
        status = self.ckpt.restore(self.checkpoint_manager.latest_checkpoint)
        # Neglict partial variables which are not restored, such as optimizer, iterator, step, etc.
        status.expect_partial()

        if self.checkpoint_manager.latest_checkpoint:
            logging.info("Restored model {} from {}".format(self.model.name, self.checkpoint_manager.latest_checkpoint))
        else:
            logging.info("There is no latest checkpoint that can be found.")

        # Restore the latest checkpoint in benchmark_checkpoint_path
        if self.benchmark_model.name != 'Interpolation':
            self.ckpt_benchmark = tf.train.Checkpoint(net=self.benchmark_model)
            self.checkpoint_manager_benchmark = tf.train.CheckpointManager(self.ckpt_benchmark, directory=self.benchmark_checkpoint_path, max_to_keep=self.ckpt_max_to_keep)
            status_benchmark = self.ckpt_benchmark.restore(self.checkpoint_manager_benchmark.latest_checkpoint)
            # Neglict partial variables which are not restored, such as optimizer, iterator, step, etc.
            status_benchmark.expect_partial()

            if self.checkpoint_manager_benchmark.latest_checkpoint:
                logging.info("Restored benchmark model {} from {}".format(self.benchmark_model.name, self.checkpoint_manager_benchmark.latest_checkpoint))
            else:
                logging.info("There is no latest checkpoint that can be found for benchmark model.")

    def eval_step(self, low_res, high_res, min, mean, maxs, test_type, loss_function_type, data_processing_combination):
        '''One step of evaluation'''
        if test_type == 'test':
            predictions = self.model(low_res, training=False)
            predictions_back = normalisation_back(predictions, maxs, mean, min, data_processing_combination)
            predictions_back_amplitude = tf.abs(predictions_back)
            high_res_amplitude = tf.abs(high_res)
            loss = self.eval_step_loss
        else:
            raise ValueError('This test type is not supported in the evaluation step function.')

        # calculate mean of loss and accuracy
        if loss_function_type == 'mse':
            loss.update_state(loss_function(high_res_amplitude, predictions_back_amplitude))
        elif loss_function_type == 'sdr':
            loss.update_state(sdr(high_res_amplitude, predictions_back_amplitude))
        elif loss_function_type == 'lsd':
            loss.update_state(lsd(high_res_amplitude, predictions_back_amplitude))
        elif loss_function_type == 'weighted_mse':
            loss.update_state(weighted_mse(high_res_amplitude, predictions_back_amplitude))
        elif loss_function_type == 'perceptual':
            loss.update_state(
                vgg_perceptual_loss(high_res_amplitude, predictions_back_amplitude, self.feature_extractor))
        else:
            raise ValueError('This loss function type is not configured in the validation step function.')

        return predictions

    def eval_benchmark_step(self, low_res, high_res):
        '''One step of evaluation'''
        predictions = self.benchmark_model(low_res, training=False)
        # calculate mean of loss and accuracy
        self.eval_benchmark_loss.update_state(loss_function(y_true=high_res, y_pred=predictions))
        return predictions

    def eval(self, global_test_max, global_test_mean, global_test_min):

        # if self.loss_function_type == 'perceptual':
        self.vgg = VGG19(weights='imagenet', include_top=False)
        self.vgg.trainable = False
        # Extract the chosen layers for feature extraction
        outputs = [self.vgg.get_layer(name).output for name in self.layer_names]
        self.feature_extractor = Model(inputs=self.vgg.input, outputs=outputs)
        self.feature_extractor.trainable = False

        logging.info("The evaluation losses are following:")
        for loss_function_type in ['mse', 'sdr', 'lsd', 'weighted_mse', 'perceptual']:
            for test_batch_idx, (test_low_res, test_high_res) in tqdm(enumerate(self.ds_test), desc='Evaluation in the epoch', leave=False, disable=False):
                test_low_res = tf.transpose(test_low_res, perm=[0, 2, 3, 1])
                test_high_res = tf.transpose(test_high_res, perm=[0, 2, 3, 1])
                test_low_res_abs_normalised, test_high_res_abs_normalised, mean, maxs, min = normalisation_processing(test_low_res, test_high_res, global_test_max, global_test_mean, global_test_min, self.processing_method)

                test_super_res = self.eval_step(test_low_res_abs_normalised, test_high_res_abs_normalised, min, mean, maxs, 'test', loss_function_type, self.processing_method)
                test_super_res = normalisation_back(test_super_res, maxs, mean, min, self.processing_method)

                test_benchmark_super_res = self.eval_step(test_low_res_abs_normalised, test_high_res_abs_normalised, min, mean, maxs, 'benchmark', loss_function_type, self.processing_method)
                test_benchmark_super_res = normalisation_back(test_benchmark_super_res, maxs, mean, min, self.processing_method)

            eval_loss = self.eval_step_loss.result().numpy()
            logging.info(f"The {loss_function_type} loss of the model is {eval_loss}")
            eval_benchmark_loss = self.eval_benchmark_loss.result().numpy()
            logging.info(f"The {loss_function_type} loss of the benchmark is {eval_benchmark_loss}")

            # Reset train metrics
            self.eval_step_loss.reset_states()
            self.eval_benchmark_loss.reset_states()


        synthesis_23_rd(test_low_res[0, :, :, -1], test_super_res[0, :, :, -1], test_high_res[0, :, :, -1], test_benchmark_super_res[0, :, :, -1], self.range_res, self.velocity_res)

        logging.info("------------------------End of the evaluation process------------------------")



# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Use for evaluating the models
# @gin.configurable
# class Evaluation():
#     def __init__(self, ds_test, model, benchmark_model, checkpoint_restore_folder, benchmark_checkpoint_path, ckpt_max_to_keep, processing_method, layer_names, whether_fftshift):
#         self.ds_test = ds_test
#         self.checkpoint_restore_folder = checkpoint_restore_folder
#         self.ckpt_max_to_keep = ckpt_max_to_keep
#         self.processing_method = processing_method
#         self.layer_names = layer_names
#         self.whether_fftshift = whether_fftshift
#
#         self.eval_step_loss = tf.keras.metrics.Mean()
#
#         self.range_res, self.velocity_res = resolution_calculation()
#
#         logging.info("------------------------Start the evaluation process------------------------")
#
#     def eval_step(self, low_res, high_res, min, mean, maxs, test_type, loss_function_type, data_processing_combination):
#         '''One step of evaluation'''
#         if test_type == 'test':
#             predictions = self.model(low_res, training=False)
#             predictions_back = normalisation_back(predictions, maxs, mean, min, data_processing_combination)
#             predictions_back_amplitude = tf.math.abs(predictions_back)
#             high_res_amplitude = tf.math.abs(high_res)
#             loss = self.eval_step_loss
#         else:
#             raise ValueError('This test type is not supported in the evaluation step function.')
#
#         # calculate mean of loss and accuracy
#         if loss_function_type == 'mse':
#             loss.update_state(loss_function(high_res_amplitude, predictions_back_amplitude))
#         elif loss_function_type == 'sdr':
#             loss.update_state(sdr(high_res_amplitude, predictions_back_amplitude))
#         elif loss_function_type == 'lsd':
#             loss.update_state(lsd(high_res_amplitude, predictions_back_amplitude))
#         elif loss_function_type == 'weighted_mse':
#             loss.update_state(weighted_mse(high_res_amplitude, predictions_back_amplitude))
#         elif loss_function_type == 'perceptual':
#             loss.update_state(
#                 vgg_perceptual_loss(high_res_amplitude, predictions_back_amplitude, self.feature_extractor))
#         else:
#             raise ValueError('This loss function type is not configured in the validation step function.')
#
#         return predictions
#
#     def eval(self, global_test_max, global_test_mean, global_test_min):
#
#         # if self.loss_function_type == 'perceptual':
#         self.vgg = VGG19(weights='imagenet', include_top=False)
#         self.vgg.trainable = False
#         # Extract the chosen layers for feature extraction
#         outputs = [self.vgg.get_layer(name).output for name in self.layer_names]
#         self.feature_extractor = Model(inputs=self.vgg.input, outputs=outputs)
#         self.feature_extractor.trainable = False
#
#         self.super_res_collection = {}
#         # data_processing_combination = 'conv&transposed&re/im&no_log&no_abs_normalization&no_angle_normalization'
#         data_processing_combination = 'padding&shuffle&ap/ph&with_log10&abs_normalization(0,1)&no_angle_normalization'
#
#         for model_name in ['Interpolation', 'CNN_simple', 'UNet_simple', 'UNet_concat', 'DP', 'SwinIR+DP', 'SwinIR+Swin']:
#         # for model_name in ['DP']:
#             logging.info(f"Start evaluating the model {model_name}")
#             if model_name == 'Interpolation':
#                 self.model = interpolation_upsampling()
#             elif model_name == 'CNN_simple':
#                 self.model = CNN_simple()
#             elif model_name == 'UNet_simple':
#                 self.model = UNet_simple()
#             elif model_name == 'UNet_concat':
#                 self.model = UNet_concat()
#             elif model_name == 'DP':
#                 self.model = DPTransformerSTFT()
#             elif model_name == 'SwinIR+DP':
#                 self.model = SwinIR(transformer_type='DP', processing_method=data_processing_combination)
#             elif model_name == 'SwinIR+Swin':
#                 self.model = SwinIR(transformer_type='Swin', processing_method=data_processing_combination)
#             elif model_name == 'DP_test':
#                 self.model = DPTransformerSTFT_test()
#             else:
#                 raise ValueError('Model name is not supported in evaluation function.')
#
#             checkpoint_restore_path = os.path.join(self.checkpoint_restore_folder, model_name)
#             # Restore the latest checkpoint in checkpoint_dir
#             if model_name != 'Interpolation':
#                 self.ckpt = tf.train.Checkpoint(net=self.model)
#                 self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, directory=checkpoint_restore_path,
#                                                                      max_to_keep=self.ckpt_max_to_keep)
#                 status = self.ckpt.restore(self.checkpoint_manager.latest_checkpoint)
#                 # Neglict partial variables which are not restored, such as optimizer, iterator, step, etc.
#                 status.expect_partial()
#
#                 if self.checkpoint_manager.latest_checkpoint:
#                     logging.info(
#                         "Restored model {} from {}".format(self.model.name, self.checkpoint_manager.latest_checkpoint))
#                 else:
#                     logging.info("There is no latest checkpoint that can be found.")
#
#             # for loss_function_type in ['mse', 'sdr', 'lsd', 'weighted_mse', 'perceptual']:
#             for loss_function_type in ['mse', 'sdr', 'lsd', 'weighted_mse']:
#                 for test_batch_idx, (test_low_res, test_high_res) in tqdm(enumerate(self.ds_test), desc='Evaluation in the epoch', leave=False, disable=False):
#                     if loss_function_type=='mse' and test_batch_idx == 1:
#                         logging.info(f"{self.model.summary()}")
#                     test_low_res = tf.transpose(test_low_res, perm=[0, 2, 3, 1])
#                     test_high_res = tf.transpose(test_high_res, perm=[0, 2, 3, 1])
#                     test_low_res_abs_normalised, test_high_res_abs_normalised, mean, maxs, min = normalisation_processing(test_low_res, test_high_res, global_test_max, global_test_mean, global_test_min, self.processing_method)
#
#                     test_super_res = self.eval_step(test_low_res_abs_normalised, test_high_res, min, mean, maxs, 'test', loss_function_type, data_processing_combination)
#                     test_super_res = normalisation_back(test_super_res, maxs, mean, min, self.processing_method)
#
#                 eval_loss = self.eval_step_loss.result().numpy()
#                 logging.info(f"The {loss_function_type} loss of the model is {eval_loss}")
#
#                 # Reset train metrics
#                 self.eval_step_loss.reset_states()
#
#             self.super_res_collection[model_name] = test_super_res[0, :, :, -1]
#
#         synthesis_33_rd(test_low_res[0, :, :, -1], test_high_res[0, :, :, -1], self.super_res_collection['Interpolation'], self.super_res_collection['CNN_simple'], self.super_res_collection['UNet_simple'], self.super_res_collection['UNet_concat'], self.super_res_collection['DP'], self.super_res_collection['SwinIR+DP'], self.super_res_collection['SwinIR+Swin'], self.range_res, self.velocity_res)
#
#         logging.info("------------------------End of the evaluation process------------------------")



# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Use for evaluating data processing combinations
# @gin.configurable
# class Evaluation():
#     def __init__(self, ds_test, model, benchmark_model, checkpoint_restore_folder, benchmark_checkpoint_path, ckpt_max_to_keep, processing_method, layer_names, whether_fftshift):
#         self.ds_test = ds_test
#         self.checkpoint_restore_folder = checkpoint_restore_folder
#         self.ckpt_max_to_keep = ckpt_max_to_keep
#         self.layer_names = layer_names
#         self.whether_fftshift = whether_fftshift
#
#         self.eval_step_loss_mse = tf.keras.metrics.Mean()
#         self.eval_step_loss_sdr = tf.keras.metrics.Mean()
#         self.eval_step_loss_lsd = tf.keras.metrics.Mean()
#         self.eval_step_loss_wmse = tf.keras.metrics.Mean()
#         self.eval_step_loss_perceptual = tf.keras.metrics.Mean()
#
#         self.range_res, self.velocity_res = resolution_calculation()
#
#         logging.info("------------------------Start the evaluation process------------------------")
#
#
#     # def eval_step(self, low_res, high_res, min, mean, maxs, test_type, loss_function_type, data_processing_combination):
#     #     '''One step of evaluation'''
#     #     if test_type == 'test':
#     #         predictions = self.model(low_res, training=False)
#     #         predictions_back = normalisation_back(predictions, maxs, mean, min, data_processing_combination)
#     #         predictions_back_amplitude = tf.abs(predictions_back)
#     #         high_res_amplitude = tf.abs(high_res)
#     #         loss = self.eval_step_loss
#     #     else:
#     #         raise ValueError('This test type is not supported in the evaluation step function.')
#     #
#     #     # calculate mean of loss and accuracy
#     #     if loss_function_type == 'mse':
#     #         loss.update_state(loss_function(high_res_amplitude, predictions_back_amplitude))
#     #     elif loss_function_type == 'sdr':
#     #         loss.update_state(sdr(high_res_amplitude, predictions_back_amplitude))
#     #     elif loss_function_type == 'lsd':
#     #         loss.update_state(lsd(high_res_amplitude, predictions_back_amplitude))
#     #     elif loss_function_type == 'weighted_mse':
#     #         loss.update_state(weighted_mse(high_res_amplitude, predictions_back_amplitude))
#     #     elif loss_function_type == 'perceptual':
#     #         loss.update_state(vgg_perceptual_loss(high_res_amplitude, predictions_back_amplitude, self.feature_extractor))
#     #     else:
#     #         raise ValueError('This loss function type is not configured in the validation step function.')
#     #
#     #     return predictions
#
#     def eval(self, global_test_max, global_test_mean, global_test_min):
#
#         # if self.loss_function_type == 'perceptual':
#         self.vgg = VGG19(weights='imagenet', include_top=False)
#         self.vgg.trainable = False
#         # Extract the chosen layers for feature extraction
#         outputs = [self.vgg.get_layer(name).output for name in self.layer_names]
#         self.feature_extractor = Model(inputs=self.vgg.input, outputs=outputs)
#         self.feature_extractor.trainable = False
#
#         self.super_res_collection = {}
#
#         # combination_types = ['A', 'B', 'C', 'D', 'E', 'F']
#         combination_types = ['E']
#
#         for combination_type in combination_types:
#             if combination_type == 'A':
#                 data_processing_combinations = ['no_processing&transposed&re/im&no_log&no_abs_normalization&no_angle_normalization', 'no_processing&transposed&ap&no_log&no_abs_normalization&no_angle_normalization', 'no_processing&transposed&ap/ph&no_log&no_abs_normalization&no_angle_normalization']
#             elif combination_type == 'B':
#                 data_processing_combinations = ['no_processing&transposed&ap/ph&no_log&no_abs_normalization&no_angle_normalization', 'padding&transposed&ap/ph&no_log&no_abs_normalization&no_angle_normalization', 'conv&transposed&ap/ph&no_log&no_abs_normalization&no_angle_normalization']
#             elif combination_type == 'B+':
#                 data_processing_combinations = ['no_processing&shuffle&ap/ph&no_log&no_abs_normalization&no_angle_normalization', 'padding&shuffle&ap/ph&no_log&no_abs_normalization&no_angle_normalization', 'conv&shuffle&ap/ph&no_log&no_abs_normalization&no_angle_normalization']
#             elif combination_type == 'C':
#                 data_processing_combinations = ['padding&transposed&ap/ph&no_log&no_abs_normalization&no_angle_normalization', 'padding&shuffle&ap/ph&no_log&no_abs_normalization&no_angle_normalization']
#             # elif combination_type == 'D':
#             #     data_processing_combinations = ['padding&shuffle&ap/ph&no_log&no_abs_normalization&no_angle_normalization', 'padding&shuffle&ap/ph&with_log2&no_abs_normalization&no_angle_normalization', 'padding&shuffle&ap/ph&with_log10&no_abs_normalization&no_angle_normalization']
#             elif combination_type == 'D':
#                 data_processing_combinations = [
#                     'padding&shuffle&ap/ph&no_log&no_abs_normalization&no_angle_normalization',
#                     'padding&shuffle&ap/ph&with_log10&no_abs_normalization&no_angle_normalization']
#             elif combination_type == 'E':
#                 data_processing_combinations = [
#                     'padding&shuffle&ap/ph&with_log10&no_abs_normalization&no_angle_normalization',
#                     'padding&shuffle&ap/ph&with_log10&abs_normalization(-1,1)&no_angle_normalization',
#                     'padding&shuffle&ap/ph&with_log10&abs_normalization(0,1)&no_angle_normalization']
#             elif combination_type == 'E+':
#                 data_processing_combinations = [
#                     'padding&shuffle&ap/ph&with_log10&no_abs_normalization&angle_normalization',
#                     'padding&shuffle&ap/ph&with_log10&abs_normalization(-1,1)&angle_normalization',
#                     'padding&shuffle&ap/ph&with_log10&abs_normalization(0,1)&angle_normalization']
#             elif combination_type == 'F':
#                 data_processing_combinations = ['padding&shuffle&ap/ph&with_log10&abs_normalization(0,1)&no_angle_normalization', 'padding&shuffle&ap/ph&with_log10&abs_normalization(0,1)&angle_normalization']
#             elif combination_type == 'F+':
#                 data_processing_combinations = ['conv&transposed&ap/ph&with_log10&abs_normalization(0,1)&no_angle_normalization', 'conv&transposed&ap/ph&with_log10&abs_normalization(0,1)&angle_normalization']
#             else:
#                 raise ValueError(f'The combination type {combination_type} is not supported in the evaluation function.')
#
#             for combination_index, data_processing_combination in enumerate(data_processing_combinations):
#                 if combination_index != 1:
#                     continue
#                 model_name = 'DP'
#                 logging.info(f"Start evaluating the data processing combination {data_processing_combination} with the model {model_name}")
#                 if model_name == 'Interpolation':
#                     self.model = interpolation_upsampling()
#                 elif model_name == 'CNN_simple':
#                     self.model = CNN_simple()
#                 elif model_name == 'UNet_simple':
#                     self.model = UNet_simple()
#                 elif model_name == 'UNet_concat':
#                     self.model = UNet_concat()
#                 elif model_name == 'DP':
#                     print(combination_index)
#                     # if combination_index != 2:
#                     self.model = DPTransformerSTFT(processing_method=data_processing_combination, dff=64, num_dp_layers = 1)
#                     # elif combination_index == 1:
#                     #     self.model = DPTransformerSTFT(processing_method=data_processing_combination, dff=96, num_dp_layers = 1)
#                     # else:
#                     #     self.model = DPTransformerSTFT(processing_method=data_processing_combination, dff=88, num_dp_layers=2)
#                 elif model_name == 'SwinIR+DP':
#                     self.model = SwinIR(transformer_type='DP', processing_method=data_processing_combination)
#                 elif model_name == 'SwinIR+Swin':
#                     self.model = SwinIR(transformer_type='Swin', processing_method=data_processing_combination)
#                 else:
#                     raise ValueError('Model name is not supported in evaluation function.')
#
#                 checkpoint_restore_path = os.path.join(self.checkpoint_restore_folder, combination_type, str(combination_index+1))
#                 # Restore the latest checkpoint in checkpoint_dir
#                 if model_name != 'Interpolation':
#                     self.ckpt = tf.train.Checkpoint(net=self.model)
#                     self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, directory=checkpoint_restore_path,
#                                                                          max_to_keep=self.ckpt_max_to_keep)
#                     status = self.ckpt.restore(self.checkpoint_manager.latest_checkpoint)
#                     # Neglict partial variables which are not restored, such as optimizer, iterator, step, etc.
#                     status.expect_partial()
#
#                     if self.checkpoint_manager.latest_checkpoint:
#                         logging.info(
#                             "Restored model {} from {}".format(self.model.name, self.checkpoint_manager.latest_checkpoint))
#                     else:
#                         logging.info("There is no latest checkpoint that can be found.")
#
#                 # for loss_function_type in ['mse', 'sdr', 'lsd', 'weighted_mse', 'perceptual']:
#                 # # for loss_function_type in ['weighted_mse']:
#                 for test_batch_idx, (test_low_res, test_high_res) in tqdm(enumerate(self.ds_test), desc='Evaluation in the epoch', leave=False, disable=False):
#                     test_low_res = tf.transpose(test_low_res, perm=[0, 2, 3, 1])
#                     test_high_res = tf.transpose(test_high_res, perm=[0, 2, 3, 1])
#                     test_low_res_abs_normalised, test_high_res_abs_normalised, mean, maxs, min = normalisation_processing(test_low_res, test_high_res, global_test_max, global_test_mean, global_test_min, data_processing_combination)
#                     # test_super_res = self.eval_step(test_low_res_abs_normalised, test_high_res, min, mean, maxs, 'test', loss_function_type, data_processing_combination)
#                     # test_super_res = normalisation_back(test_super_res, maxs, mean, min, data_processing_combination)
#                     # if test_batch_idx == 1:
#                     #     logging.info(f"{self.model.summary()}")
#                     predictions = self.model(test_low_res_abs_normalised, training=False)
#                     predictions_back = normalisation_back(predictions, maxs, mean, min, data_processing_combination)
#                     predictions_back_amplitude = tf.math.abs(predictions_back)
#                     high_res_amplitude = tf.math.abs(test_high_res)
#
#                     self.eval_step_loss_mse.update_state(loss_function(high_res_amplitude, predictions_back_amplitude))
#                     self.eval_step_loss_sdr.update_state(sdr(high_res_amplitude, predictions_back_amplitude))
#                     self.eval_step_loss_lsd.update_state(lsd(high_res_amplitude, predictions_back_amplitude))
#                     self.eval_step_loss_wmse.update_state(weighted_mse(high_res_amplitude, predictions_back_amplitude))
#                     self.eval_step_loss_perceptual.update_state(vgg_perceptual_loss(high_res_amplitude, predictions_back_amplitude, self.feature_extractor))
#
#                 eval_loss_mse = self.eval_step_loss_mse.result().numpy()
#                 logging.info(f"The MSE loss of the model is {eval_loss_mse}")
#                 eval_loss_sdr = self.eval_step_loss_sdr.result().numpy()
#                 logging.info(f"The SDR loss of the model is {eval_loss_sdr}")
#                 eval_loss_lsd = self.eval_step_loss_lsd.result().numpy()
#                 logging.info(f"The LSD loss of the model is {eval_loss_lsd}")
#                 eval_loss_wmse = self.eval_step_loss_wmse.result().numpy()
#                 logging.info(f"The WMSE loss of the model is {eval_loss_wmse}")
#                 eval_loss_perceptual = self.eval_step_loss_perceptual.result().numpy()
#                 logging.info(f"The Perceptual loss of the model is {eval_loss_perceptual}")
#
#                 # Reset train metrics
#                 self.eval_step_loss_mse.reset_states()
#                 self.eval_step_loss_sdr.reset_states()
#                 self.eval_step_loss_lsd.reset_states()
#                 self.eval_step_loss_wmse.reset_states()
#                 self.eval_step_loss_perceptual.reset_states()
#
#                 name = str(combination_type + str(combination_index+1))
#                 self.super_res_collection[name] = predictions_back_amplitude[0, :, :, -1]
#
#             if combination_index == 2:
#                 synthesis_13_super_rd(list(self.super_res_collection.values())[0], list(self.super_res_collection.values())[1], list(self.super_res_collection.values())[2], self.range_res, self.velocity_res)
#             elif combination_index == 1:
#                 synthesis_12_super_rd(list(self.super_res_collection.values())[0], list(self.super_res_collection.values())[1], self.range_res, self.velocity_res)
#         logging.info("------------------------End of the evaluation process------------------------")


# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Use for evaluating loss functions combinations
# @gin.configurable
# class Evaluation():
#     def __init__(self, ds_test, model, benchmark_model, checkpoint_restore_folder, benchmark_checkpoint_path, ckpt_max_to_keep, processing_method, layer_names, whether_fftshift):
#         self.ds_test = ds_test
#         self.checkpoint_restore_folder = checkpoint_restore_folder
#         self.ckpt_max_to_keep = ckpt_max_to_keep
#         self.layer_names = layer_names
#         self.whether_fftshift = whether_fftshift
#
#         self.eval_step_loss_mse = tf.keras.metrics.Mean()
#         self.eval_step_loss_sdr = tf.keras.metrics.Mean()
#         self.eval_step_loss_lsd = tf.keras.metrics.Mean()
#         self.eval_step_loss_wmse = tf.keras.metrics.Mean()
#         self.eval_step_loss_perceptual = tf.keras.metrics.Mean()
#
#         self.range_res, self.velocity_res = resolution_calculation()
#
#         logging.info("------------------------Start the evaluation process------------------------")
#
#
#     def eval(self, global_test_max, global_test_mean, global_test_min):
#
#         # if self.loss_function_type == 'perceptual':
#         self.vgg = VGG19(weights='imagenet', include_top=False)
#         self.vgg.trainable = False
#         # Extract the chosen layers for feature extraction
#         outputs = [self.vgg.get_layer(name).output for name in self.layer_names]
#         self.feature_extractor = Model(inputs=self.vgg.input, outputs=outputs)
#         self.feature_extractor.trainable = False
#
#         self.super_res_collection = {}
#
#         data_processing_combination = 'padding&shuffle&ap/ph&with_log10&abs_normalization(0,1)&no_angle_normalization'
#         # data_processing_combination = 'padding&shuffle&ap/ph&no_log&no_abs_normalization&no_angle_normalization'
#
#         # loss_function_types = ['mse', 'sdr', 'lsd', 'plsd', 'wmse', 'perceptual']
#         # loss_function_types = ['lsd']
#         # loss_function_types = ['mse+vgg', 'lsd+vgg', 'plsd+vgg', 'wmse+vgg']
#         # loss_function_types = ['lambda=1', 'lambda=0.5', 'lambda=0.1', 'lambda=1e-2', 'lambda=5e-3', 'lambda=1e-3']
#         loss_function_types = ['lambda=1', 'lambda=5e-1', 'lambda=1e-1', 'lambda=1e-2', 'lambda=1e-3', 'lambda=1e-4']
#         # loss_function_types = ['lambda=5e-1']
#         # loss_function_types = ['b1c2+b2c2+b3c4', 'b1c2+b2c2+b3c4+b4c4', 'b1c2+b2c2+b3c4+b4c4+b5c4', 'b2c2+b3c4+b4c4']
#
#         for loss_function_type in loss_function_types:
#             logging.info(f"Start evaluating the DP model trained with the loss function {loss_function_type}")
#             self.model = DPTransformerSTFT()
#             checkpoint_restore_path = os.path.join(self.checkpoint_restore_folder, loss_function_type)
#             # Restore the latest checkpoint in checkpoint_dir
#             self.ckpt = tf.train.Checkpoint(net=self.model)
#             self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, directory=checkpoint_restore_path,
#                                                                  max_to_keep=self.ckpt_max_to_keep)
#             status = self.ckpt.restore(self.checkpoint_manager.latest_checkpoint)
#             # Neglict partial variables which are not restored, such as optimizer, iterator, step, etc.
#             status.expect_partial()
#
#             if self.checkpoint_manager.latest_checkpoint:
#                 logging.info(
#                     "Restored model {} from {}".format(self.model.name, self.checkpoint_manager.latest_checkpoint))
#             else:
#                 logging.info("There is no latest checkpoint that can be found.")
#
#             # for loss_function_type in ['mse', 'sdr', 'lsd', 'weighted_mse', 'perceptual']:
#             # # for loss_function_type in ['weighted_mse']:
#             for test_batch_idx, (test_low_res, test_high_res) in tqdm(enumerate(self.ds_test), desc='Evaluation in the epoch', leave=False, disable=False):
#                 test_low_res = tf.transpose(test_low_res, perm=[0, 2, 3, 1])
#                 test_high_res = tf.transpose(test_high_res, perm=[0, 2, 3, 1])
#                 test_low_res_abs_normalised, test_high_res_abs_normalised, mean, maxs, min = normalisation_processing(test_low_res, test_high_res, global_test_max, global_test_mean, global_test_min, data_processing_combination)
#
#                 # test_super_res = self.eval_step(test_low_res_abs_normalised, test_high_res, min, mean, maxs, 'test', loss_function_type, data_processing_combination)
#                 # test_super_res = normalisation_back(test_super_res, maxs, mean, min, data_processing_combination)
#                 predictions = self.model(test_low_res_abs_normalised, training=False)
#                 predictions_back = normalisation_back(predictions, maxs, mean, min, data_processing_combination)
#                 predictions_back_amplitude = tf.math.abs(predictions_back)
#                 high_res_amplitude = tf.math.abs(test_high_res)
#
#                 self.eval_step_loss_mse.update_state(loss_function(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_sdr.update_state(sdr(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_lsd.update_state(lsd(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_wmse.update_state(weighted_mse(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_perceptual.update_state(vgg_perceptual_loss(high_res_amplitude, predictions_back_amplitude, self.feature_extractor))
#
#             eval_loss_mse = self.eval_step_loss_mse.result().numpy()
#             logging.info(f"The MSE loss of the model is {eval_loss_mse}")
#             eval_loss_sdr = self.eval_step_loss_sdr.result().numpy()
#             logging.info(f"The SDR loss of the model is {eval_loss_sdr}")
#             eval_loss_lsd = self.eval_step_loss_lsd.result().numpy()
#             logging.info(f"The LSD loss of the model is {eval_loss_lsd}")
#             eval_loss_wmse = self.eval_step_loss_wmse.result().numpy()
#             logging.info(f"The WMSE loss of the model is {eval_loss_wmse}")
#             eval_loss_perceptual = self.eval_step_loss_perceptual.result().numpy()
#             logging.info(f"The Perceptual loss of the model is {eval_loss_perceptual}")
#
#             # Reset train metrics
#             self.eval_step_loss_mse.reset_states()
#             self.eval_step_loss_sdr.reset_states()
#             self.eval_step_loss_lsd.reset_states()
#             self.eval_step_loss_wmse.reset_states()
#             self.eval_step_loss_perceptual.reset_states()
#
#             self.super_res_collection[str(loss_function_type)] = predictions_back_amplitude[0, :, :, -1]
#
#         if len(loss_function_types) == 6:
#             synthesis_23_super_rd(list(self.super_res_collection.values())[0], list(self.super_res_collection.values())[1], list(self.super_res_collection.values())[2], list(self.super_res_collection.values())[3], list(self.super_res_collection.values())[4], list(self.super_res_collection.values())[5], self.range_res, self.velocity_res)
#         elif len(loss_function_types) == 4:
#             synthesis_22_super_rd(list(self.super_res_collection.values())[0], list(self.super_res_collection.values())[1], list(self.super_res_collection.values())[2], list(self.super_res_collection.values())[3], self.range_res, self.velocity_res)
#         elif len(loss_function_types) == 1:
#             synthesis_13_rd(tf.math.abs(test_low_res)[0, :, :, -1], list(self.super_res_collection.values())[0], high_res_amplitude[0, :, :, -1], self.range_res, self.velocity_res)
#         else:
#             raise ValueError('The number of loss function types is not supported the visualization in the evaluation function.')
#
#         logging.info("------------------------End of the evaluation process------------------------")


# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Use for evaluating the differences of the architectures and transformer blocks
# @gin.configurable
# class Evaluation():
#     def __init__(self, ds_test, model, benchmark_model, checkpoint_restore_folder, benchmark_checkpoint_path,
#                  ckpt_max_to_keep, processing_method, layer_names, whether_fftshift):
#         self.ds_test = ds_test
#         self.checkpoint_restore_folder = checkpoint_restore_folder
#         self.ckpt_max_to_keep = ckpt_max_to_keep
#         self.layer_names = layer_names
#         self.whether_fftshift = whether_fftshift
#
#         self.eval_step_loss_mse = tf.keras.metrics.Mean()
#         self.eval_step_loss_sdr = tf.keras.metrics.Mean()
#         self.eval_step_loss_lsd = tf.keras.metrics.Mean()
#         self.eval_step_loss_wmse = tf.keras.metrics.Mean()
#         self.eval_step_loss_perceptual = tf.keras.metrics.Mean()
#
#         self.range_res, self.velocity_res = resolution_calculation()
#
#         logging.info("------------------------Start the evaluation process------------------------")
#
#     def eval(self, global_test_max, global_test_mean, global_test_min):
#
#         # if self.loss_function_type == 'perceptual':
#         self.vgg = VGG19(weights='imagenet', include_top=False)
#         self.vgg.trainable = False
#         # Extract the chosen layers for feature extraction
#         outputs = [self.vgg.get_layer(name).output for name in self.layer_names]
#         self.feature_extractor = Model(inputs=self.vgg.input, outputs=outputs)
#         self.feature_extractor.trainable = False
#
#         self.super_res_collection = {}
#
#         data_processing_combination = 'padding&shuffle&ap/ph&with_log10&abs_normalization(0,1)&no_angle_normalization'
#
#         # conditions_combinations = {'0': 'DP'}
#         # conditions_combinations = {'0+': 'SwinIR+DP'}
#         conditions_combinations = {'13': [False, False, False, False, False, False, False, True, True, False, False, False, False]}
#
#         # conditions_combinations = {'1':[True, False, False, False, False, False, False, False, False, False, False, False],
#         #                            '2':[True, True, False, False, False, False, False, False, False, False, False, False],
#         #                            '3':[True, True, True, False, False, False, False, False, False, False, False, False],
#         #                            '4':[True, True, True, True, False, False, False, False, False, False, False, False],
#         #                            '5':[True, True, True, True, True, False, False, False, False, False, False, False],
#         #                            '6':[True, True, True, True, True, True, False, False, False, False, False, False]}
#
#         # conditions_combinations = {'7':[True, True, True, True, True, True, True, False, False, False, False, False],
#         #                            '8':[True, True, True, True, True, True, True, True, False, False, False, False],
#         #                            '9':[True, True, True, True, True, True, True, True, True, False, False, False],
#         #                            '10':[True, True, True, True, True, True, True, True, True, True, False, False]}
#
#         # conditions_combinations = {'7':[True, True, True, True, True, True, True, False, False, False, False, False, False],
#         #                            '8':[True, True, True, True, True, True, True, True, False, False, False, False, False],
#         #                            '9':[True, True, True, True, True, True, True, True, True, False, False, False, False],
#         #                            '10':[True, True, True, True, True, True, True, True, True, True, False, False, False],
#         #                            '11':[True, True, True, True, True, True, True, True, True, True, True, False, False],
#         #                            '12':[True, True, True, True, True, True, True, True, True, True, True, False, True]}
#
#         for conditions_combination_index, conditions_combination in conditions_combinations.items():
#             logging.info(f"Start evaluating the DP_test model trained with the conditions combination as {conditions_combination}")
#             if conditions_combination_index == '0':
#                 self.model = DPTransformerSTFT()
#             elif conditions_combination_index == '0+':
#                 self.model = SwinIR(transformer_type='DP', processing_method=data_processing_combination)
#             else:
#                 self.model = DPTransformerSTFT_test(conditions=conditions_combination)
#             checkpoint_restore_path = os.path.join(self.checkpoint_restore_folder, conditions_combination_index)
#             # Restore the latest checkpoint in checkpoint_dir
#             self.ckpt = tf.train.Checkpoint(net=self.model)
#             self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, directory=checkpoint_restore_path, max_to_keep=self.ckpt_max_to_keep)
#             status = self.ckpt.restore(self.checkpoint_manager.latest_checkpoint)
#             # Neglict partial variables which are not restored, such as optimizer, iterator, step, etc.
#             status.expect_partial()
#
#             if self.checkpoint_manager.latest_checkpoint:
#                 logging.info(
#                     "Restored model {} from {}".format(self.model.name, self.checkpoint_manager.latest_checkpoint))
#             else:
#                 logging.info("There is no latest checkpoint that can be found.")
#
#             # for loss_function_type in ['mse', 'sdr', 'lsd', 'weighted_mse', 'perceptual']:
#             # # for loss_function_type in ['weighted_mse']:
#             for test_batch_idx, (test_low_res, test_high_res) in tqdm(enumerate(self.ds_test), desc='Evaluation in the epoch', leave=False, disable=False):
#                 test_low_res = tf.transpose(test_low_res, perm=[0, 2, 3, 1])
#                 test_high_res = tf.transpose(test_high_res, perm=[0, 2, 3, 1])
#                 test_low_res_abs_normalised, test_high_res_abs_normalised, mean, maxs, min = normalisation_processing(
#                     test_low_res, test_high_res, global_test_max, global_test_mean, global_test_min,
#                     data_processing_combination)
#                 if test_batch_idx == 1:
#                     logging.info(f"{self.model.summary()}")
#                 # test_super_res = self.eval_step(test_low_res_abs_normalised, test_high_res, min, mean, maxs, 'test', loss_function_type, data_processing_combination)
#                 # test_super_res = normalisation_back(test_super_res, maxs, mean, min, data_processing_combination)
#                 predictions = self.model(test_low_res_abs_normalised, training=False)
#                 predictions_back = normalisation_back(predictions, maxs, mean, min, data_processing_combination)
#                 predictions_back_amplitude = tf.math.abs(predictions_back)
#                 high_res_amplitude = tf.math.abs(test_high_res)
#
#                 self.eval_step_loss_mse.update_state(loss_function(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_sdr.update_state(sdr(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_lsd.update_state(lsd(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_wmse.update_state(weighted_mse(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_perceptual.update_state(
#                     vgg_perceptual_loss(high_res_amplitude, predictions_back_amplitude, self.feature_extractor))
#
#             eval_loss_mse = self.eval_step_loss_mse.result().numpy()
#             logging.info(f"The MSE loss of the model is {eval_loss_mse}")
#             eval_loss_sdr = self.eval_step_loss_sdr.result().numpy()
#             logging.info(f"The SDR loss of the model is {eval_loss_sdr}")
#             eval_loss_lsd = self.eval_step_loss_lsd.result().numpy()
#             logging.info(f"The LSD loss of the model is {eval_loss_lsd}")
#             eval_loss_wmse = self.eval_step_loss_wmse.result().numpy()
#             logging.info(f"The WMSE loss of the model is {eval_loss_wmse}")
#             eval_loss_perceptual = self.eval_step_loss_perceptual.result().numpy()
#             logging.info(f"The Perceptual loss of the model is {eval_loss_perceptual}")
#
#             # Reset train metrics
#             self.eval_step_loss_mse.reset_states()
#             self.eval_step_loss_sdr.reset_states()
#             self.eval_step_loss_lsd.reset_states()
#             self.eval_step_loss_wmse.reset_states()
#             self.eval_step_loss_perceptual.reset_states()
#
#             self.super_res_collection[str(conditions_combination_index)] = predictions_back_amplitude[0, :, :, -1]
#
#         if len(conditions_combinations) == 6:
#             synthesis_23_super_rd(list(self.super_res_collection.values())[0],
#                                   list(self.super_res_collection.values())[1],
#                                   list(self.super_res_collection.values())[2],
#                                   list(self.super_res_collection.values())[3],
#                                   list(self.super_res_collection.values())[4],
#                                   list(self.super_res_collection.values())[5], self.range_res, self.velocity_res)
#         elif len(conditions_combinations) == 4:
#             synthesis_22_super_rd(list(self.super_res_collection.values())[0],
#                                   list(self.super_res_collection.values())[1],
#                                   list(self.super_res_collection.values())[2],
#                                   list(self.super_res_collection.values())[3], self.range_res, self.velocity_res)
#         elif len(conditions_combinations) == 1:
#             synthesis_13_super_rd(tf.math.abs(test_low_res)[0, :, :, -1], predictions_back_amplitude[0, :, :, -1], high_res_amplitude[0, :, :, -1], self.range_res, self.velocity_res)
#
#         logging.info("------------------------End of the evaluation process------------------------")


# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Use for evaluating cGAN
# @gin.configurable
# class Evaluation():
#     def __init__(self, ds_test, model, benchmark_model, checkpoint_restore_folder, benchmark_checkpoint_path, ckpt_max_to_keep, processing_method, layer_names, whether_fftshift):
#         self.ds_test = ds_test
#         self.checkpoint_restore_folder = checkpoint_restore_folder
#         self.ckpt_max_to_keep = ckpt_max_to_keep
#         self.layer_names = layer_names
#         self.whether_fftshift = whether_fftshift
#
#         self.eval_step_loss_mse = tf.keras.metrics.Mean()
#         self.eval_step_loss_sdr = tf.keras.metrics.Mean()
#         self.eval_step_loss_lsd = tf.keras.metrics.Mean()
#         self.eval_step_loss_wmse = tf.keras.metrics.Mean()
#         self.eval_step_loss_perceptual = tf.keras.metrics.Mean()
#
#         self.range_res, self.velocity_res = resolution_calculation()
#
#         logging.info("------------------------Start the evaluation process------------------------")
#
#
#     def eval(self, global_test_max, global_test_mean, global_test_min):
#
#         # if self.loss_function_type == 'perceptual':
#         self.vgg = VGG19(weights='imagenet', include_top=False)
#         self.vgg.trainable = False
#         # Extract the chosen layers for feature extraction
#         outputs = [self.vgg.get_layer(name).output for name in self.layer_names]
#         self.feature_extractor = Model(inputs=self.vgg.input, outputs=outputs)
#         self.feature_extractor.trainable = False
#
#         self.super_res_collection = {}
#
#         data_processing_combination = 'padding&shuffle&ap/ph&with_log10&abs_normalization(0,1)&no_angle_normalization'
#
#         # loss_function_types = ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
#         # for no low-resolution discriminator
#         loss_function_types = ['lambda=5e-1', 'lambda=1e-1', 'lambda=5e-2', 'lambda=1e-2', 'lambda=5e-3', 'lambda=1e-3']
#         # for with low-resolution discriminator
#         # loss_function_types = ['lambda=1e-1', 'lambda=5e-2', 'lambda=1e-2', 'lambda=5e-3', 'lambda=1e-3', 'lambda=1e-4']
#
#         for loss_function_type in loss_function_types:
#             logging.info(f"Start evaluating the cGAN model.")
#             self.model = ConditionalGAN()
#             checkpoint_restore_path = os.path.join(self.checkpoint_restore_folder, loss_function_type)
#             # Restore the latest checkpoint in checkpoint_dir
#             self.ckpt = tf.train.Checkpoint(net=self.model.generator)
#             self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, directory=checkpoint_restore_path,
#                                                                  max_to_keep=self.ckpt_max_to_keep)
#             status = self.ckpt.restore(self.checkpoint_manager.latest_checkpoint)
#             # Neglict partial variables which are not restored, such as optimizer, iterator, step, etc.
#             status.expect_partial()
#
#             if self.checkpoint_manager.latest_checkpoint:
#                 logging.info(
#                     "Restored model {} from {}".format(self.model.name, self.checkpoint_manager.latest_checkpoint))
#             else:
#                 logging.info("There is no latest checkpoint that can be found.")
#
#             # for loss_function_type in ['mse', 'sdr', 'lsd', 'weighted_mse', 'perceptual']:
#             # # for loss_function_type in ['weighted_mse']:
#             for test_batch_idx, (test_low_res, test_high_res) in tqdm(enumerate(self.ds_test), desc='Evaluation in the epoch', leave=False, disable=False):
#                 test_low_res = tf.transpose(test_low_res, perm=[0, 2, 3, 1])
#                 test_high_res = tf.transpose(test_high_res, perm=[0, 2, 3, 1])
#                 test_low_res_abs_normalised, test_high_res_abs_normalised, mean, maxs, min = normalisation_processing(test_low_res, test_high_res, global_test_max, global_test_mean, global_test_min, data_processing_combination)
#
#                 # test_super_res = self.eval_step(test_low_res_abs_normalised, test_high_res, min, mean, maxs, 'test', loss_function_type, data_processing_combination)
#                 # test_super_res = normalisation_back(test_super_res, maxs, mean, min, data_processing_combination)
#                 predictions = self.model(inputs=test_low_res_abs_normalised, type='generator', training=False)
#                 predictions_back = normalisation_back(predictions, maxs, mean, min, data_processing_combination)
#                 predictions_back_amplitude = tf.math.abs(predictions_back)
#                 high_res_amplitude = tf.math.abs(test_high_res)
#
#                 self.eval_step_loss_mse.update_state(loss_function(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_sdr.update_state(sdr(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_lsd.update_state(lsd(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_wmse.update_state(weighted_mse(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_perceptual.update_state(vgg_perceptual_loss(high_res_amplitude, predictions_back_amplitude, self.feature_extractor))
#
#             eval_loss_mse = self.eval_step_loss_mse.result().numpy()
#             logging.info(f"The MSE loss of the model is {eval_loss_mse}")
#             eval_loss_sdr = self.eval_step_loss_sdr.result().numpy()
#             logging.info(f"The SDR loss of the model is {eval_loss_sdr}")
#             eval_loss_lsd = self.eval_step_loss_lsd.result().numpy()
#             logging.info(f"The LSD loss of the model is {eval_loss_lsd}")
#             eval_loss_wmse = self.eval_step_loss_wmse.result().numpy()
#             logging.info(f"The WMSE loss of the model is {eval_loss_wmse}")
#             eval_loss_perceptual = self.eval_step_loss_perceptual.result().numpy()
#             logging.info(f"The Perceptual loss of the model is {eval_loss_perceptual}")
#
#             # Reset train metrics
#             self.eval_step_loss_mse.reset_states()
#             self.eval_step_loss_sdr.reset_states()
#             self.eval_step_loss_lsd.reset_states()
#             self.eval_step_loss_wmse.reset_states()
#             self.eval_step_loss_perceptual.reset_states()
#
#             self.super_res_collection[str(loss_function_type)] = predictions_back_amplitude[0, :, :, -1]
#
#         if len(loss_function_types) == 6:
#             synthesis_23_super_rd(list(self.super_res_collection.values())[0], list(self.super_res_collection.values())[1], list(self.super_res_collection.values())[2], list(self.super_res_collection.values())[3], list(self.super_res_collection.values())[4], list(self.super_res_collection.values())[5], self.range_res, self.velocity_res)
#         elif len(loss_function_types) == 4:
#             synthesis_22_super_rd(list(self.super_res_collection.values())[0], list(self.super_res_collection.values())[1], list(self.super_res_collection.values())[2], list(self.super_res_collection.values())[3], self.range_res, self.velocity_res)
#         elif len(loss_function_types) == 1:
#             synthesis_13_rd(tf.math.abs(test_low_res)[0, :, :, -1], list(self.super_res_collection.values())[0], high_res_amplitude[0, :, :, -1], self.range_res, self.velocity_res)
#         else:
#             raise ValueError('The number of loss function types is not supported the visualization in the evaluation function.')
#
#         logging.info("------------------------End of the evaluation process------------------------")



# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Use for evaluating FOL or evaluating resampling rate
# @gin.configurable
# class Evaluation():
#     def __init__(self, ds_test, model, benchmark_model, checkpoint_restore_folder, benchmark_checkpoint_path, ckpt_max_to_keep, processing_method, layer_names, whether_fftshift):
#         self.ds_test = ds_test
#         self.checkpoint_restore_folder = checkpoint_restore_folder
#         self.ckpt_max_to_keep = ckpt_max_to_keep
#         self.layer_names = layer_names
#         self.whether_fftshift = whether_fftshift
#
#         self.eval_step_loss_mse = tf.keras.metrics.Mean()
#         self.eval_step_loss_sdr = tf.keras.metrics.Mean()
#         self.eval_step_loss_lsd = tf.keras.metrics.Mean()
#         self.eval_step_loss_wmse = tf.keras.metrics.Mean()
#         self.eval_step_loss_perceptual = tf.keras.metrics.Mean()
#
#         self.range_res, self.velocity_res = resolution_calculation()
#
#         logging.info("------------------------Start the evaluation process------------------------")
#
#
#     def eval(self, global_test_max, global_test_mean, global_test_min):
#
#         # if self.loss_function_type == 'perceptual':
#         self.vgg = VGG19(weights='imagenet', include_top=False)
#         self.vgg.trainable = False
#         # Extract the chosen layers for feature extraction
#         outputs = [self.vgg.get_layer(name).output for name in self.layer_names]
#         self.feature_extractor = Model(inputs=self.vgg.input, outputs=outputs)
#         self.feature_extractor.trainable = False
#
#         self.super_res_collection = {}
#
#         data_processing_combination = 'padding&shuffle&ap/ph&with_log10&abs_normalization(0,1)&no_angle_normalization'
#
#         training_types = ['DP', 'cGAN']
#
#         for training_type in training_types:
#             if training_type == 'cGAN':
#                 logging.info(f"Start evaluating the cGAN model.")
#                 self.model = ConditionalGAN()
#                 checkpoint_restore_path = os.path.join(self.checkpoint_restore_folder, training_type)
#                 # Restore the latest checkpoint in checkpoint_dir
#                 self.ckpt = tf.train.Checkpoint(net=self.model.generator)
#                 self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, directory=checkpoint_restore_path,
#                                                                      max_to_keep=self.ckpt_max_to_keep)
#                 status = self.ckpt.restore(self.checkpoint_manager.latest_checkpoint)
#                 # Neglict partial variables which are not restored, such as optimizer, iterator, step, etc.
#                 status.expect_partial()
#
#                 if self.checkpoint_manager.latest_checkpoint:
#                     logging.info(
#                         "Restored model {} from {}".format(self.model.name, self.checkpoint_manager.latest_checkpoint))
#                 else:
#                     logging.info("There is no latest checkpoint that can be found.")
#             elif training_type == 'DP':
#                 logging.info(f"Start evaluating the DP model.")
#                 self.model = DPTransformerSTFT()
#                 checkpoint_restore_path = os.path.join(self.checkpoint_restore_folder, training_type)
#                 # Restore the latest checkpoint in checkpoint_dir
#                 self.ckpt = tf.train.Checkpoint(net=self.model)
#                 self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, directory=checkpoint_restore_path, max_to_keep=self.ckpt_max_to_keep)
#                 status = self.ckpt.restore(self.checkpoint_manager.latest_checkpoint)
#                 # Neglict partial variables which are not restored, such as optimizer, iterator, step, etc.
#                 status.expect_partial()
#
#                 if self.checkpoint_manager.latest_checkpoint:
#                     logging.info(
#                         "Restored model {} from {}".format(self.model.name, self.checkpoint_manager.latest_checkpoint))
#                 else:
#                     logging.info("There is no latest checkpoint that can be found.")
#             else:
#                 raise ValueError('The training type is not supported in the evaluation function.')
#
#             # for loss_function_type in ['mse', 'sdr', 'lsd', 'weighted_mse', 'perceptual']:
#             # # for loss_function_type in ['weighted_mse']:
#             for test_batch_idx, (test_low_res, test_high_res) in tqdm(enumerate(self.ds_test), desc='Evaluation in the epoch', leave=False, disable=False):
#                 test_low_res = tf.transpose(test_low_res, perm=[0, 2, 3, 1])
#                 test_high_res = tf.transpose(test_high_res, perm=[0, 2, 3, 1])
#                 test_low_res_abs_normalised, test_high_res_abs_normalised, mean, maxs, min = normalisation_processing(test_low_res, test_high_res, global_test_max, global_test_mean, global_test_min, data_processing_combination)
#
#                 # test_super_res = self.eval_step(test_low_res_abs_normalised, test_high_res, min, mean, maxs, 'test', loss_function_type, data_processing_combination)
#                 # test_super_res = normalisation_back(test_super_res, maxs, mean, min, data_processing_combination)
#                 if training_type == 'cGAN':
#                     predictions = self.model(inputs=test_low_res_abs_normalised, type='generator', training=False)
#                 elif training_type == 'DP':
#                     predictions = self.model(test_low_res_abs_normalised, training=False)
#                 else:
#                     raise ValueError('The training type is not supported in the evaluation loop function.')
#                 predictions_back = normalisation_back(predictions, maxs, mean, min, data_processing_combination)
#                 predictions_back_amplitude = tf.math.abs(predictions_back)
#                 high_res_amplitude = tf.math.abs(test_high_res)
#
#                 self.eval_step_loss_mse.update_state(loss_function(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_sdr.update_state(sdr(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_lsd.update_state(lsd(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_wmse.update_state(weighted_mse(high_res_amplitude, predictions_back_amplitude))
#                 self.eval_step_loss_perceptual.update_state(vgg_perceptual_loss(high_res_amplitude, predictions_back_amplitude, self.feature_extractor))
#
#             eval_loss_mse = self.eval_step_loss_mse.result().numpy()
#             logging.info(f"The MSE loss of the model is {eval_loss_mse}")
#             eval_loss_sdr = self.eval_step_loss_sdr.result().numpy()
#             logging.info(f"The SDR loss of the model is {eval_loss_sdr}")
#             eval_loss_lsd = self.eval_step_loss_lsd.result().numpy()
#             logging.info(f"The LSD loss of the model is {eval_loss_lsd}")
#             eval_loss_wmse = self.eval_step_loss_wmse.result().numpy()
#             logging.info(f"The WMSE loss of the model is {eval_loss_wmse}")
#             eval_loss_perceptual = self.eval_step_loss_perceptual.result().numpy()
#             logging.info(f"The Perceptual loss of the model is {eval_loss_perceptual}")
#
#             # Reset train metrics
#             self.eval_step_loss_mse.reset_states()
#             self.eval_step_loss_sdr.reset_states()
#             self.eval_step_loss_lsd.reset_states()
#             self.eval_step_loss_wmse.reset_states()
#             self.eval_step_loss_perceptual.reset_states()
#
#             self.super_res_collection[str(training_type)] = predictions_back_amplitude[0, :, :, -1]
#
#         if len(training_types) == 2:
#             synthesis_22_rd(tf.math.abs(test_low_res)[0, :, :, -1], high_res_amplitude[0, :, :, -1], list(self.super_res_collection.values())[0], list(self.super_res_collection.values())[1], self.range_res, self.velocity_res)
#         else:
#             raise ValueError('The number of loss function types is not supported the visualization in the evaluation function.')
#
#         logging.info("------------------------End of the evaluation process------------------------")