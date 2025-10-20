from itertools import cycle

import gin
import logging
import tensorflow as tf
import time
from tqdm import tqdm
import os
import wandb
import numpy as np
from packaging import version
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

from models.architectures import *
from models.layers import re_im_separation
from evaluation.visualization import plot_range_doppler, resolution_calculation
from loss_functions import loss_function, sd_sdr, sdr, lsd, generator_loss, discriminator_loss, plsd, weighted_mse, vgg_perceptual_loss
from utils.data_processing import normalisation_processing, normalisation_back
from models.cGAN import Discriminator


@gin.configurable
class Trainer(object):
    def __init__(self, logger, strategy, ds_train, ds_val, global_train_max, global_train_mean, global_train_min, global_val_max, global_val_mean, global_val_min, model, model_name, checkpoint_path, task_type, learning_rate, epochs, ckpt_max_to_keep, optimizer_type, early_stop_patience, early_stop_type, loss_type, lr_schedule_type, overfitting_type, layer_names, lambda_lsd, lambda_vgg, lambda_disc, processing_method, sampling_rate, pretraining=False):
        self.logger = logger
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.global_train_max = global_train_max
        self.global_train_mean = global_train_mean
        self.global_train_min = global_train_min
        self.global_val_max = global_val_max
        self.global_val_mean = global_val_mean
        self.global_val_min = global_val_min
        self.model = model
        self.model_name = model_name
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.strategy = strategy
        # To use the data subset, remember to change the decay step in the learning rate schedule
        if lr_schedule_type:
            self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=10000,
                decay_rate=0.96,
                staircase=True
            )
        # Load the optimizer
        with self.strategy.scope():
            if version.parse(tf.__version__) >= version.parse("2.11"):
                # Available in tf 2.11 and later, and fast on Apple Silicon
                optimizers_module = tf.keras.optimizers.legacy
            else:
                # On remote server, the old tf version doesn't support the legacy optimizers
                optimizers_module = tf.keras.optimizers
            self.optimizer_class = getattr(optimizers_module, self.optimizer_type)
            if lr_schedule_type:
                self.optimizer = self.optimizer_class(learning_rate=self.lr_schedule)
            else:
                self.optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.ckpt_max_to_keep = ckpt_max_to_keep
        self.task_type = task_type
        self.early_stop_patience = early_stop_patience
        self.early_stop_type = early_stop_type
        self.loss_type = loss_type
        self.loss_function_type, self.data_type = self.loss_type.split('&')
        self.overfitting_type = overfitting_type
        self.layer_names = layer_names
        self.lambda_lsd = lambda_lsd
        self.lambda_vgg = lambda_vgg
        self.lambda_disc = lambda_disc
        self.processing_method = processing_method
        self.sampling_rate = sampling_rate
        self.pretraining = pretraining
        # Split all the processing methods
        self.processing_type, self.upsampling_type, self.separation_type, self.log_type, self.abs_normalization_type, self.angle_normalization_type = self.processing_method.split('&')
        # Initialize the feature extractor in the VGG perceptual loss which will be updated in the train function
        self.feature_extractor = None
        # Calculate the range and velocity resolution
        self.range_res, self.velocity_res = resolution_calculation()
        # Initialize the mean operation of train and validation loss
        self.train_step_loss = tf.keras.metrics.Mean()
        self.val_step_loss = tf.keras.metrics.Mean()

        # Set up the checkpoint
        with self.strategy.scope():
            iterator = iter(self.ds_train)
            # By default, the step starts with 1 while epoch begins with 0
            step = tf.Variable(1, name='step')
            epoch = tf.Variable(0, name='epoch')
            if self.model_name != 'cGAN':
                self.ckpt = tf.train.Checkpoint(step=step, epoch=epoch, optimizer=self.optimizer, net=self.model, iterator=iterator)
            else:
                self.ckpt = tf.train.Checkpoint(step=step, epoch=epoch, optimizer=self.optimizer, net=self.model.generator, iterator=iterator)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=self.ckpt_max_to_keep)

            # wandb.log({'Type of optimizer': self.optimizer_type})

    def log_model_summary(self):
        summary_lines = []
        self.model.summary(expand_nested=True, print_fn=lambda x: summary_lines.append(x))
        for line in summary_lines:
            self.logger.info(line)

    def seek_amplitude(self, data):
        # Seek the amplitude of the data
        if self.separation_type == 're/im':
            # return tf.math.sqrt(tf.math.square(data[0, :, :, 0]) + tf.math.square(data[0, :, :, 1]))
            return abs(data[0, :, :, 0])
        elif self.separation_type == 'ap' or self.separation_type == 'ap/ph':
            return abs(data[0, :, :, 0])
        else:
            raise ValueError('The separation type is not supported in the seeking amplitude function.')

    @tf.function
    def train_step(self, low_res, high_res, min, mean, max, epoch):
        def train_step_fn(low_res, high_res, min, mean, max, epoch):
            # One step of training
            with tf.GradientTape() as tape:
                # training=False is only needed if there are layers with different behavior during training versus inference (e.g. Dropout).
                super_res = self.model(low_res, training=True)
                if self.loss_function_type == 'mse':
                    mse_loss = loss_function(high_res, super_res, self.data_type)
                    vgg_loss = vgg_perceptual_loss(high_res, super_res, self.feature_extractor)
                    train_loss = (self.lambda_lsd * mse_loss + self.lambda_vgg * vgg_loss) / (self.lambda_lsd + self.lambda_vgg)
                elif self.loss_function_type == 'sd_sdr':
                    # train_loss = calc_min_perm_loss(super_res, high_res, self.loss_function_type)
                    train_loss = sd_sdr(high_res, super_res, self.data_type)
                elif self.loss_function_type == 'sdr':
                    if epoch < 50:
                        train_loss, mask, masked_gt = lsd(high_res, super_res, self.data_type, min, mean, max)
                    else:
                        sdr_loss = sdr(high_res, super_res, self.data_type)
                        vgg_loss = vgg_perceptual_loss(high_res, super_res, self.feature_extractor)
                        train_loss = (self.lambda_lsd * sdr_loss + self.lambda_vgg * vgg_loss) / (self.lambda_lsd + self.lambda_vgg)
                elif self.loss_function_type == 'lsd':
                    # If combining with the perceptual loss, LSD will pre-train the model in the first 50 epochs
                    if self.pretraining:
                        if epoch < 50:
                            train_loss, mask, masked_gt = lsd(high_res, super_res, self.data_type, min, mean, max)
                        else:
                            lsd_loss, mask, masked_gt = lsd(high_res, super_res, self.data_type, min, mean, max)
                            vgg_loss = vgg_perceptual_loss(high_res, super_res, self.feature_extractor)
                            train_loss = (self.lambda_lsd * lsd_loss + self.lambda_vgg * vgg_loss) / (self.lambda_lsd + self.lambda_vgg)
                    else:
                        lsd_loss, mask, masked_gt = lsd(high_res, super_res, self.data_type, min, mean, max)
                        vgg_loss = vgg_perceptual_loss(high_res, super_res, self.feature_extractor)
                        train_loss = (self.lambda_lsd * lsd_loss + self.lambda_vgg * vgg_loss) / (self.lambda_lsd + self.lambda_vgg)
                    # else:
                    #     cycle_epoch = epoch % 10
                    #     if cycle_epoch < 8:
                    #         train_loss, mask, masked_gt = lsd(high_res, super_res, self.data_type, mean)
                    #     else:
                    #         train_loss = vgg_perceptual_loss(high_res, super_res, self.feature_extractor)
                elif self.loss_function_type == 'plsd':
                    plsd_loss = plsd(high_res, super_res)
                    vgg_loss = vgg_perceptual_loss(high_res, super_res, self.feature_extractor)
                    train_loss = (self.lambda_lsd * plsd_loss + self.lambda_vgg * vgg_loss) / (self.lambda_lsd + self.lambda_vgg)
                elif self.loss_function_type == 'weighted_mse':
                    wmse_loss = weighted_mse(high_res, super_res, min, mean, max, self.data_type)
                    vgg_loss = vgg_perceptual_loss(high_res, super_res, self.feature_extractor)
                    train_loss = (self.lambda_lsd * wmse_loss + self.lambda_vgg * vgg_loss) / (self.lambda_lsd + self.lambda_vgg)
                elif self.loss_function_type == 'perceptual':
                    # If training by the perceptual loss, the LSD will pre-train the model in the first 50 epochs for the stability
                    if self.pretraining:
                        if epoch < 50:
                            train_loss, mask, masked_gt = lsd(high_res, super_res, self.data_type, min, mean, max)
                        else:
                            train_loss = vgg_perceptual_loss(high_res, super_res, self.feature_extractor)
                    else:
                        vgg_loss = vgg_perceptual_loss(high_res, super_res, self.feature_extractor)
                        lsd_loss, mask, masked_gt = lsd(high_res, super_res, self.data_type, min, mean, max)

                        # disc_real_output = self.model2(inputs=high_res, type='discriminator', training=True)
                        # disc_generated_output = self.model2(inputs=super_res, type='discriminator', training=True)
                        # gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(self.loss_function_type, disc_generated_output, super_res, high_res, epoch, self.feature_extractor)
                        # disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

                        # train_loss = (self.lambda_lsd * lsd_loss + self.lambda_vgg * vgg_loss + self.lambda_disc * gen_gan_loss) / (self.lambda_lsd + self.lambda_vgg + self.lambda_disc)
                        train_loss = (self.lambda_lsd * lsd_loss + self.lambda_vgg * vgg_loss) / (self.lambda_lsd + self.lambda_vgg)
                else:
                    raise ValueError('This loss function type is not configured in the training step function.')

            # calculate the gradient between loss and variables of model
            gradients = tape.gradient(train_loss, self.model.trainable_variables)
            gradients = [tf.convert_to_tensor(g) if isinstance(g, tf.IndexedSlices) else g for g in gradients]
            # apply gradients with learning rate on variables of model
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            # discriminator_gradients = disc_tape.gradient(disc_loss, self.model2.discriminator.trainable_variables)
            # self.optimizer.apply_gradients(zip(discriminator_gradients, self.model2.discriminator.trainable_variables))

            # return train_loss, super_res, mask, masked_gt
            return train_loss, super_res

        def train_step_cgan(low_res, high_res, min, mean, max, epoch, convertBack=False):
            # If the trained model is the cGAN model, then this train step works
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                super_res = self.model(inputs=low_res, type='generator', training=True)

                if convertBack:
                    high_res_unnormalised = high_res * max
                    high_res_unnormalised = tf.math.pow(10.0, high_res_unnormalised)

                    super_res_unnormalised = super_res * max
                    super_res_unnormalised = tf.math.pow(10.0, super_res_unnormalised)

                # # For the case of the discriminator with low-resolution data as the condition
                # disc_real_output = self.model(inputs=tf.expand_dims(low_res[:, :, :, -1], axis=-1), type='discriminator', targets=high_res, training=True)
                # disc_generated_output = self.model(inputs=tf.expand_dims(low_res[:, :, :, -1], axis=-1), type='discriminator', targets=super_res, training=True)

                # For the case of the discriminator without low-resolution data as the condition
                disc_real_output = self.model(inputs=high_res, type='discriminator', training=True)
                disc_generated_output = self.model(inputs=super_res, type='discriminator', training=True)

                # gen_total_loss, gen_gan_loss, gen_l1_loss, mask, masked_gt = generator_loss(disc_generated_output, super_res, high_res, epoch)
                gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(self.loss_function_type, disc_generated_output, super_res, high_res, epoch, self.feature_extractor, min, mean, max)
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

            generator_gradients = gen_tape.gradient(gen_total_loss, self.model.generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss, self.model.discriminator.trainable_variables)

            self.optimizer.apply_gradients(zip(generator_gradients, self.model.generator.trainable_variables))
            self.optimizer.apply_gradients(zip(discriminator_gradients, self.model.discriminator.trainable_variables))

            # return gen_total_loss, super_res, mask, masked_gt
            return gen_total_loss, super_res

        if self.model_name != 'cGAN':
            per_replica_output = self.strategy.run(train_step_fn, args=(low_res, high_res, min, mean, max, epoch, ))
            # In the output per replica, only the loss is needed to be meaned while the super-resolution data is concatenated
            per_replica_losses, per_replica_super_res = per_replica_output
            total_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            # total_loss, per_replica_super_res = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        else:
            per_replica_output = self.strategy.run(train_step_cgan, args=(low_res, high_res, min, mean, max, epoch, ))
            per_replica_losses, per_replica_super_res = per_replica_output
            total_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            # # total_loss, per_replica_super_res, per_replica_mask, per_replica_masked_gt = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            # total_loss, per_replica_super_res = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        self.train_step_loss.update_state(total_loss)
        super_res_all = tf.concat(self.strategy.experimental_local_results(per_replica_super_res), axis=0)
        # mask = tf.concat(self.strategy.experimental_local_results(per_replica_mask), axis=0)
        # masked_gt = tf.concat(self.strategy.experimental_local_results(per_replica_masked_gt), axis=0)
        # return super_res_all, mask, masked_gt
        return super_res_all

    # def train_step(self, low_res, high_res):
    #     # One step of training
    #     with tf.GradientTape() as tape:
    #         # training=False is only needed if there are layers with different behavior during training versus inference (e.g. Dropout).
    #         super_res = self.model(low_res, training=True)
    #         train_loss = loss_function(high_res, super_res)
    #     # calculate the gradient between loss and variables of model
    #     gradients = tape.gradient(train_loss, self.model.trainable_variables)
    #     # apply gradients with learning rate on variables of model
    #     self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    #     # calculate mean of loss and accuracy
    #     self.train_step_loss.update_state(train_loss)
    #     return super_res

    def val_step(self, low_res, high_res, min, mean, max):
        '''One step of validation'''
        if self.model_name != 'cGAN':
            predictions = self.model(low_res, training=True)

            # calculate mean of loss and accuracy
            if self.loss_function_type == 'mse':
                self.val_step_loss.update_state(loss_function(high_res, predictions, self.data_type))
            elif self.loss_function_type == 'sd_sdr':
                # self.val_step_loss.update_state(calc_min_perm_loss(predictions, high_res, self.loss_function_type))
                self.val_step_loss.update_state(sd_sdr(high_res, predictions, self.data_type))
            elif self.loss_function_type == 'sdr':
                self.val_step_loss.update_state(sdr(high_res, predictions, self.data_type))
            elif self.loss_function_type == 'lsd':
                val_loss, val_mask, val_masked_gt = lsd(high_res, predictions, self.data_type, min, mean, max)
                self.val_step_loss.update_state(val_loss)
            elif self.loss_function_type == 'plsd':
                self.val_step_loss.update_state(plsd(high_res, predictions))
            elif self.loss_function_type == 'weighted_mse':
                self.val_step_loss.update_state(weighted_mse(high_res, predictions, min, mean, max, self.data_type))
            elif self.loss_function_type == 'perceptual':
                self.val_step_loss.update_state(vgg_perceptual_loss(high_res, predictions, self.feature_extractor))
            else:
                raise ValueError('This loss function type is not configured in the validation step function.')
        else:
            predictions = self.model(inputs=low_res, type='generator', training=True)

            # disc_real_output = self.model(inputs=high_res, type='discriminator', training=True)
            # disc_generated_output = self.model(inputs=predictions, type='discriminator', training=True)
            # val_loss = discriminator_loss(disc_real_output, disc_generated_output)
            if self.loss_function_type == 'lsd':
                # Use the loss combination as the validation loss
                lsd_loss, val_mask, val_masked_gt = lsd(high_res, predictions, self.data_type, min, mean, max)
                vgg_loss = vgg_perceptual_loss(high_res, predictions, self.feature_extractor)
                val_loss = lsd_loss + 0.5 * vgg_loss
            else:
                raise ValueError('This loss function type is not configured in the validation step.')
            self.val_step_loss.update_state(val_loss)

        return predictions

    def train(self):
        '''The pipeline for the training process'''

        # # Restore the latest checkpoint in checkpoint_dir
        # self.ckpt.restore(self.manager.latest_checkpoint)
        # if self.manager.latest_checkpoint:
        #     print("Restored from {}".format(self.manager.latest_checkpoint))
        # else:
        #     print("Initializing from scratch.")

        self.logger.info('------------------------Start the training process------------------------')
        # Record the number of epochs in which the validation loss doesn't decrease
        wait = 0

        # Load the pre-trained VGG model as the feature extractor for the perceptual loss
        # if self.loss_function_type == 'perceptual':
        self.vgg = VGG19(weights='imagenet', include_top=False)
        self.vgg.trainable = False
        # Extract the chosen layers for feature extraction
        outputs = [self.vgg.get_layer(name).output for name in self.layer_names]
        self.feature_extractor = Model(inputs=self.vgg.input, outputs=outputs)
        self.feature_extractor.trainable = False

        for epoch in range(self.epochs):
            self.logger.info(f"Start of epoch {epoch + 1}")
            # save the initial time of this epoch
            start_time = time.time()

            # Reset train metrics
            self.train_step_loss.reset_states()

            for batch_idx, (low_res, high_res) in tqdm(enumerate(self.ds_train), desc='Training in the epoch', leave=False, disable=True):
                # Transpose operation to make the channel dimension as the last dimension
                train_low_res = tf.transpose(low_res, perm=[0, 2, 3, 1])
                train_high_res = tf.transpose(high_res, perm=[0, 2, 3, 1])
                # Processing methods applied on the low- and high-resolution data
                low_res_abs_normalised, high_res_abs_normalised, mean, maxs, min = normalisation_processing(train_low_res, train_high_res, self.global_train_max, self.global_train_mean, self.global_train_min, self.processing_method)
                # train_super_res = self.train_step(train_low_res, train_high_res)
                train_super_res = self.train_step(low_res_abs_normalised, high_res_abs_normalised, min, mean, maxs, epoch)
                # Convert the processing methods back to the original format
                train_super_res = normalisation_back(train_super_res, maxs, mean, min, self.processing_method)

                # Obtain the #Params information of the model
                if epoch == 0 and batch_idx == 0:
                    # self.model.build(input_shape=(2, 129, 32, 4))
                    if self.model_name != 'cGAN':
                        self.logger.info(f"{self.model.summary()}")
                    else:
                        self.logger.info(f"{self.model.generator.summary()}")
                        self.logger.info(f"{self.model.discriminator.summary()}")
                    # self.logger.info(f"{self.model.summary(expand_nested=True)}")
                    # self.log_model_summary()
                    self.logger.handlers[0].flush()
                # Update the index of steps
                self.ckpt.step.assign_add(1)
            # Update the index of epochs
            self.ckpt.epoch.assign_add(1)

            # If pre-training, no validation is needed
            if self.pretraining and epoch < 50:
                template = 'epoch {}, Train Loss: {}, Time taken: {}.'
                self.logger.info(template.format(epoch + 1,
                                                 self.train_step_loss.result().numpy(),
                                                 time.time() - start_time
                                                 )
                                 )
                if self.task_type != 'Tune':
                    wandb.log({'Train low resolutional image': wandb.Image(plot_range_doppler((self.seek_amplitude(train_low_res)) ** 2, self.range_res * 2, self.velocity_res * 2, data_type='Low resolutional', range_limit=True, scale='log')),
                               'Train super resolutional image': wandb.Image(plot_range_doppler((self.seek_amplitude(train_super_res)) ** 2, self.range_res, self.velocity_res, data_type='Super resolutional', range_limit=True, scale='log')),
                               'Train high resolutional image': wandb.Image(plot_range_doppler((self.seek_amplitude(train_high_res)) ** 2, self.range_res, self.velocity_res, data_type='High resolutional', range_limit=True, scale='log')),
                               'Train loss': self.train_step_loss.result().numpy(),
                               "Learning rate": self.optimizer._decayed_lr(tf.float32).numpy(),
                               "Time taken": time.time() - start_time
                               })
            else:
                if not self.overfitting_type:
                    # Reset val metrics
                    self.val_step_loss.reset_states()

                    for val_batch_idx, (val_low_res, val_high_res) in tqdm(enumerate(self.ds_val), desc='Validation in the epoch', leave=False, disable=True):
                        val_low_res = tf.transpose(val_low_res, perm=[0, 2, 3, 1])
                        val_high_res = tf.transpose(val_high_res, perm=[0, 2, 3, 1])
                        val_low_res_abs_normalised, val_high_res_abs_normalised, val_mean, val_maxs, val_min = normalisation_processing(val_low_res, val_high_res, self.global_val_max, self.global_val_mean, self.global_val_min, self.processing_method)
                        # val_super_res = self.val_step(val_low_res, val_high_res)
                        val_super_res = self.val_step(val_low_res_abs_normalised, val_high_res_abs_normalised, val_min, val_mean, val_maxs)
                        val_super_res = normalisation_back(val_super_res, val_maxs, val_mean, val_min, self.processing_method)

                    val_loss = self.val_step_loss.result().numpy()

                    # In the first epoch, save the loss value as the baseline of the validation loss
                    if self.pretraining:
                        # If pre-training, here initializes the validation loss
                        if epoch == 50:
                            self.val_loss_min = val_loss
                            self.best_epoch_idx = epoch
                            self.manager.save()
                    else:
                        if epoch == 0:
                            self.val_loss_min = val_loss
                            self.best_epoch_idx = epoch
                            self.manager.save()
                    # If the validation loss is less than the previous, save the checkpoint
                    if val_loss <= self.val_loss_min:
                        wait = 0
                        self.val_loss_min = val_loss
                        self.best_epoch_idx = epoch
                        save_path = self.manager.save()
                        self.logger.info("Saved checkpoint path for step {} in epoch {}: {}".format(int(self.ckpt.step.numpy()), int(self.ckpt.epoch.numpy()), save_path))
                    else:
                        wait += 1

                    template = 'epoch {}, Train Loss: {}, Validation Loss: {}, Time taken: {}, The best epoch {} has the minimum validation loss {}'
                    self.logger.info(template.format(epoch + 1,
                                                     self.train_step_loss.result().numpy(),
                                                     self.val_step_loss.result().numpy(),
                                                     time.time() - start_time,
                                                     self.best_epoch_idx + 1,
                                                     self.val_loss_min
                                                     )
                                     )

                    # Write variables to wandb
                    if self.task_type != 'Tune':
                        wandb.log({'Train low resolutional image': wandb.Image(plot_range_doppler((self.seek_amplitude(train_low_res)) ** 2, self.range_res * self.sampling_rate, self.velocity_res * self.sampling_rate, data_type='Low resolutional', range_limit=True, scale='log')),
                                   'Train super resolutional image': wandb.Image(plot_range_doppler((self.seek_amplitude(train_super_res)) ** 2, self.range_res, self.velocity_res, data_type='Super resolutional', range_limit=True, scale='log')),
                                   'Train high resolutional image': wandb.Image(plot_range_doppler((self.seek_amplitude(train_high_res)) ** 2, self.range_res, self.velocity_res, data_type='High resolutional', range_limit=True, scale='log')),
                                   'Validation low resolutional image': wandb.Image(plot_range_doppler((self.seek_amplitude(val_low_res)) ** 2, self.range_res * self.sampling_rate, self.velocity_res * self.sampling_rate, data_type='Low resolutional', range_limit=True, scale='log')),
                                   'Validation super resolutional image': wandb.Image(plot_range_doppler((self.seek_amplitude(val_super_res)) ** 2, self.range_res, self.velocity_res, data_type='Super resolutional', range_limit=True, scale='log')),
                                   'Validation high resolutional image': wandb.Image(plot_range_doppler((self.seek_amplitude(val_high_res)) ** 2, self.range_res, self.velocity_res, data_type='High resolutional', range_limit=True, scale='log')),
                                   # 'Ground truth in loss function': wandb.Image(plot_range_doppler((abs(high_res_abs_normalised[0, :, :, -1])) ** 2, self.range_res, self.velocity_res, data_type='High normalised resolutional', range_limit=False, scale='log')),
                                   # 'Mask in loss function': wandb.Image(plot_range_doppler((abs(mask[0, :, :, -1])) ** 2, self.range_res, self.velocity_res, data_type='Mask', range_limit=False, scale='log')),
                                   # 'Masked GT in loss function': wandb.Image(plot_range_doppler((abs(masked_gt[0, :, :, -1])) ** 2, self.range_res, self.velocity_res, data_type='Masked high resolutional', range_limit=False, scale='log')),
                                   'Train loss': self.train_step_loss.result().numpy(),
                                   "Validation loss": val_loss,
                                   "Best validation loss": self.val_loss_min,
                                   "Learning rate": self.optimizer._decayed_lr(tf.float32).numpy(),
                                   "Time taken": time.time() - start_time
                                   })
                    else:
                        wandb.log({'Train loss': self.train_step_loss.result().numpy(),
                                   "Validation loss": val_loss,
                                   "Best validation loss": self.val_loss_min,
                                   "Learning rate": self.optimizer._decayed_lr(tf.float32).numpy(),
                                   "Time taken": time.time() - start_time})

                # If overfitting, only the training loss is needed and no consideration on the early stop patience
                else:
                    train_loss = self.train_step_loss.result().numpy()
                    if epoch == 0:
                        self.train_loss_min = train_loss
                        self.best_epoch_idx = epoch
                        self.manager.save()

                    if train_loss <= self.train_loss_min:
                        self.train_loss_min = train_loss
                        self.best_epoch_idx = epoch
                        save_path = self.manager.save()
                        self.logger.info("Saved checkpoint path for step {} in epoch {}: {}".format(int(self.ckpt.step.numpy()), int(self.ckpt.epoch.numpy()), save_path))

                    template = 'epoch {}, Train Loss: {}, Time taken: {}, The best epoch {} has the minimum training loss {}'
                    self.logger.info(template.format(epoch + 1,
                                                 self.train_step_loss.result().numpy(),
                                                 time.time() - start_time,
                                                 self.best_epoch_idx + 1,
                                                 self.train_loss_min
                                                 )
                                 )

                    # Write variables to wandb
                    if self.task_type != 'Tune':
                        wandb.log({'Train low resolutional image': wandb.Image(plot_range_doppler((self.seek_amplitude(train_low_res))**2, self.range_res * 2, self.velocity_res * 2, data_type='Low resolutional', scale='log')),
                                   'Train super resolutional image': wandb.Image(plot_range_doppler((self.seek_amplitude(train_super_res))**2, self.range_res, self.velocity_res, data_type='Super resolutional', scale='log')),
                                   'Train high resolutional image': wandb.Image(plot_range_doppler((self.seek_amplitude(train_high_res))**2, self.range_res, self.velocity_res, data_type='High resolutional', scale='log')),
                                   'Train loss': self.train_step_loss.result().numpy(),
                                   "Best training loss": self.train_loss_min,
                                   "Learning rate": self.optimizer._decayed_lr(tf.float32).numpy(),
                                   "Time taken": time.time() - start_time
                                   })

                    # wandb.log({'Train low resolutional image': wandb.Image(plot_range_doppler((abs(tf.complex(tf.math.real(train_low_res[0, :, :, -1]) * low_max[0, :, :, -1], tf.math.imag(train_low_res[0, :, :, -1]) * low_max[0, :, :, -1]))) ** 2, self.range_res * 2, self.velocity_res * 2, data_type='Low resolutional', scale='log')),
                    #            'Train super resolutional image': wandb.Image(plot_range_doppler((abs(tf.complex(tf.math.real(train_super_res[0, :, :, -1]) * high_max[0, :, :, -1], tf.math.real(train_super_res[0, :, :, -1]) * high_max[0, :, :, -1]))) ** 2, self.range_res, self.velocity_res, data_type='Super resolutional', scale='log')),
                    #            'Train high resolutional image': wandb.Image(plot_range_doppler((abs(tf.complex(tf.math.real(train_high_res[0, :, :, -1]) * high_max[0, :, :, -1], tf.math.real(train_high_res[0, :, :, -1]) * high_max[0, :, :, -1]))) ** 2, self.range_res, self.velocity_res, data_type='High resolutional', scale='log')),
                    #            'Validation low resolutional image': wandb.Image(plot_range_doppler((abs(tf.complex(tf.math.real(val_low_res[0, :, :, -1]) * val_low_max[0, :, :, -1], tf.math.real(val_low_res[0, :, :, -1]) * val_low_max[0, :, :, -1]))) ** 2, self.range_res * 2, self.velocity_res * 2, data_type='Low resolutional', scale='log')),
                    #            'Validation super resolutional image': wandb.Image(plot_range_doppler((abs(tf.complex(tf.math.real(val_super_res[0, :, :, -1]) * val_high_max[0, :, :, -1], tf.math.real(val_super_res[0, :, :, -1]) * val_high_max[0, :, :, -1]))) ** 2, self.range_res, self.velocity_res, data_type='Super resolutional', scale='log')),
                    #            'Validation high resolutional image': wandb.Image(plot_range_doppler((abs(tf.complex(tf.math.real(val_high_res[0, :, :, -1]) * val_high_max[0, :, :, -1], tf.math.real(val_high_res[0, :, :, -1]) * val_high_max[0, :, :, -1]))) ** 2, self.range_res, self.velocity_res, data_type='High resolutional', scale='log')),
                    #            'Train loss': self.train_step_loss.result().numpy(),
                    #            "Validation loss": val_loss,
                    #            "Best validation loss": self.val_loss_min,
                    #            "Learning rate": self.optimizer._decayed_lr(tf.float32).numpy(),
                    #            "Time taken": time.time() - start_time
                    #            })

            if self.early_stop_type:
                if wait >= self.early_stop_patience:
                    self.logger.info(f"------------------------Early stopping at epoch {epoch + 1}------------------------")
                    break

        # wandb.finish()
        self.logger.info('------------------------End of the training process------------------------')
