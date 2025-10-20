import gin
import logging
from absl import app
import wandb
import argparse

from utils.datasets import sd_dataset
from train import *
from models.architectures import *
from evaluation.evaluation import Evaluation
from tune import Tuner
from tensorflow.keras.models import Model

from models.dp_transformer_stft_exponential import DPTransformerSTFT
from models.dp_test import DPTransformerSTFT_test
from models.swinir_tensorflow import SwinIR
from models.cGAN import ConditionalGAN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# if disable the wandb, uncomment the following command
# os.environ["WANDB_MODE"] = "disabled"

# replace the key with the own wandb login key of the project
wandb.login(key='cf09500bbcbf4968e54b72844ea7b9cdc2674e8d')

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.set_visible_devices(gpus, 'GPU')
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print(f"Using {len(gpus)} GPUs")
#     except RuntimeError as e:
#         print(e)
# else:
#     print("No GPU available")


def parse_args():
    # Overwrite the default gin configurations from the batch file
    parser = argparse.ArgumentParser(description="Run training with Gin configuration.")
    parser.add_argument("--gin_file", type=str, default="utils/config.gin", help="Path to the Gin configuration file.")
    parser.add_argument("--gin_bind", type=str, action="append", help="Dynamic Gin bindings to override the configuration.")
    return parser.parse_args()


def prepare_checkpoint_path(checkpoint_path, model_name, t, type):
    """
    Create a checkpoint path only for this model type
    Args:
        checkpoint_path (str): Path of the folder to save all checkpoints.
        model_name (str): Name of the model.
        t (time.struct_time): Current time.
        type (str): Type of the training ('Train', 'Test', 'Tune').

    Returns:
        checkpoint_path (str): Path of the checkpoint folder for this running job.
        checkpoint_info (dict): Information about the checkpoint.

    """
    checkpoint_path = checkpoint_path + 'ckpt_' + model_name + '/'
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    # Distinguish the checkpoint path by the start time of training
    checkpoint_info = {'model_name': model_name, 'train_time': t}
    checkpoint_path = checkpoint_path + type + '_' + str(t.tm_year) + '_' + str(t.tm_mon).zfill(2) + '_' + str(t.tm_mday).zfill(2) + '_' + str(t.tm_hour).zfill(2) + '_' + str(t.tm_min).zfill(2) + '_' + str(t.tm_sec).zfill(2) + '/'
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    return checkpoint_path, checkpoint_info


def choose_model_with_name(model_name, strategy):
    """
    According to the model name to load the model
    Args:
        model_name (str): Name of the model.
        strategy: MirroredStrategy for distributed training.

    Returns:
        model: The model object.

    """

    if model_name == 'CNN_simple':
        # # wandb.log({'Model name': model_name})
        # CNN_simple_num_layers = gin.query_parameter('CNN_simple.num_layers')
        # wandb.log({'Number of layers': CNN_simple_num_layers})
        # for i in range(CNN_simple_num_layers):
        #     CNN_simple_neuron_list = gin.query_parameter('CNN_simple.neuron_list')
        #     wandb.log({'Number of neurons in layer'+str(i+1): CNN_simple_neuron_list[i]})

        # Use strategy to load the model for the distributed training
        with strategy.scope():
            model = CNN_simple()

    elif model_name == 'UNet_simple':
        with strategy.scope():
            model = UNet_simple()

    elif model_name == 'UNet_concat':
        with strategy.scope():
            model = UNet_concat()

    elif model_name == 'Interpolation':
        with strategy.scope():
            model = interpolation_upsampling()

    elif model_name == 'DP-Transformer':
        with strategy.scope():
            model = DPTransformerSTFT()

    elif model_name == 'DP-Transformer_test':
        with strategy.scope():
            model = DPTransformerSTFT_test()

    elif model_name == 'SwinIR+Swin':
        with strategy.scope():
            model = SwinIR(transformer_type='Swin')

    elif model_name == 'SwinIR+DP':
        with strategy.scope():
            model = SwinIR(transformer_type='DP')

    elif model_name == 'cGAN':
        with strategy.scope():
            model = ConditionalGAN()

    elif model_name == 'Interpolation':
        with strategy.scope():
            model = interpolation_upsampling()

    else:
        raise ValueError('Model name is not supported.')

    # with strategy.scope():
    #     model2 = ConditionalGAN()

    # model.build(input_shape=(32, 129, 32, 4))

    return model


@gin.configurable
def main(create_tfrecords, type, checkpoint_path, model_name, benchmark_model_name, tfrecord_suffix):
    t = time.localtime()

    logger.info('-------------Hyperparameters setting---------------')
    logger.info(f"{gin.config_str()}")
    logger.info('---------------------------------------------------')

    # Create a MirroredStrategy for the distributed training
    strategy = tf.distribute.MirroredStrategy()
    logger.info("-----------Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Only when training or tuning, the new checkpoint path will be created and log file will be saved
    if type == 'Train' or type == 'Tune':
        checkpoint_path_new, checkpoint_info = prepare_checkpoint_path(checkpoint_path, model_name, t, type)

        # Save the logging infos in the checkpoint subfolder
        file_log = checkpoint_path_new + 'train.log'
        fh = logging.FileHandler(filename=file_log, encoding='utf-8', mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        fh.flush()
        logger.addHandler(fh)

    if type == 'Tune':
        tuner = Tuner(model_name, logger, checkpoint_path_new)
        tuner.tune_hyper()

    else:
        # Set up the WandB, it's also possible to add 'notes' as str or 'tags' as list in the init function
        wandb.init(
            project='MA',
            name=model_name + '_' + str(t.tm_year) + '_' + str(t.tm_mon).zfill(2) + '_' + str(t.tm_mday).zfill(2)
                 + '_' + str(t.tm_hour).zfill(2) + '_' + str(t.tm_min).zfill(2) + '_' + str(t.tm_sec).zfill(2),
        )

        # setup pipeline and load the dataset
        ds = sd_dataset(logger)
        # For creating TFRecord files
        if create_tfrecords:
            ds.creating_action(tfrecord_suffix)
        # Loading dataset from the TFRecord files
        ds_train, ds_val, ds_test, global_train_max, global_train_mean, global_train_min, global_val_max, global_val_mean, global_val_min, global_test_max, global_test_mean, global_test_min = ds.load_tfrecord(tfrecord_suffix)

        # # Show the shape of the values
        # for low_res, high_res in ds_train.take(3):
        #     print(f"Low resolutional data shape: {low_res.shape}")
        #     print(f"Low resolutional data type: {low_res.dtype}")
        #     print(f"High resolutional data shape: {high_res.shape}")
        #     print(f"High resolutional data type: {high_res.dtype}")

        model = choose_model_with_name(model_name, strategy)
        # if model_name == 'DP-Tranformer':
        #     logger.info(f"{model.model().summary(expand_nested=True)}")
        # elif model_name != 'SwinIR':
        #     logger.info(f"{model.summary(expand_nested=True)}")
        # logger.handlers[0].flush()
        logger.info(f"The model {model.name} is loaded successfully")

        if type == 'Test':
            benchmark_model = choose_model_with_name(benchmark_model_name, strategy)
            # logger.info(benchmark_model.summary())
            logger.info(f"The model {model.name} is loaded successfully")

        # train model
        if type == 'Train':
            trainer = Trainer(logger, strategy, ds_train, ds_val, global_train_max, global_train_mean, global_train_min, global_val_max, global_val_mean, global_val_min, model, model_name, checkpoint_path_new, 'Train')
            trainer.train()

            # Save all the used gin-configurable parameters
            gin_config_str = gin.operative_config_str()
            logger.info("GIN Configurable Parameters:\n%s", gin_config_str)

        elif type == 'Test':
            evaluator = Evaluation(ds_test, model, benchmark_model)
            evaluator.eval(global_test_max, global_test_mean, global_test_min)

        else:
            logger.error("Please choose the correct action: 'Train', 'Test' or 'Tune'")



if __name__ == "__main__":
    # Clear the cache of the previous configurations
    gin.clear_config()

    # #  setup gin-config
    # gin.parse_config_file('utils/config.gin')

    # Parse command line arguments
    args = parse_args()

    # Load Gin configuration file and apply bindings
    gin.parse_config_file(args.gin_file)
    if args.gin_bind:
        gin.parse_config(args.gin_bind)


    main()