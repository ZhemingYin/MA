from pprint import pprint
import wandb
import gin

from train import Trainer
from utils.datasets import sd_dataset
from models.architectures import *
from models.dp_transformer_stft_exponential import DPTransformerSTFT
from models.cGAN import ConditionalGAN


def choose_model_with_name(model_name, strategy):
    '''Load model for the tuning process'''
    if model_name == 'CNN_simple':
        # wandb.log({'Model name': model_name})
        CNN_simple_num_layers = gin.query_parameter('CNN_simple.num_layers')
        print("-----------------------------------CNN simple num of layers is ", CNN_simple_num_layers)
        wandb.log({'Number of layers': CNN_simple_num_layers})
        for i in range(CNN_simple_num_layers):
            CNN_simple_neuron_list = gin.query_parameter('CNN_simple.neuron_list')
            wandb.log({'Number of neurons in layer'+str(i+1): CNN_simple_neuron_list[i]})

        with strategy.scope():
            model = CNN_simple()

    elif model_name == 'UNet_simple':
        with strategy.scope():
            model = UNet_simple()

    elif model_name == 'UNet_concat':
        with strategy.scope():
            model = UNet_concat()

    elif model_name == 'DP-Transformer':
        with strategy.scope():
            model = DPTransformerSTFT()

    elif model_name == 'cGAN':
        with strategy.scope():
            model = ConditionalGAN()

    return model


class Tuner():
    def __init__(self, model_name, logger, checkpoint_path_new):
        self.model_name = model_name
        self.logger = logger
        self.checkpoint_path_new = checkpoint_path_new

        # Reload the gin config file
        gin.clear_config()

        # Create a MirroredStrategy.
        self.strategy = tf.distribute.MirroredStrategy()
        logger.info("-----------Number of devices: {}".format(self.strategy.num_replicas_in_sync))


    def train_func(self):
        with wandb.init() as run:

            gin.clear_config()
            # Hyperparameters
            bindings = []
            for key, value in run.config.items():
                if key == 'Trainer.optimizer_type' and isinstance(value, str):
                    value = f'"{value}"'
                bindings.append(f'{key}={value}')

            # gin-config, reload the specified gin file
            gin_file = 'utils/config_tune_' + self.model_name + '.gin'
            gin.parse_config_files_and_bindings([gin_file], bindings)

            # setup pipeline
            ds = sd_dataset(self.logger)
            # ds.creating_action()
            self.ds_train, self.ds_val, self.ds_test, global_train_max, global_train_mean, global_train_min, global_val_max, global_val_mean, global_val_min, global_test_max, global_test_mean, global_test_min = ds.load_tfrecord(tfrecord_suffix = '_factor2')

            self.model = choose_model_with_name(self.model_name, self.strategy)
            # self.logger.info(self.model.summary())
            self.logger.info('The model is loaded successfully')

            # Train with the chosen parameters in this sweep iteration
            trainer = Trainer(self.logger, self.strategy, self.ds_train, self.ds_val, global_train_max, global_train_mean, global_train_min,
                              global_val_max, global_val_mean, global_val_min, self.model, self.model_name, self.checkpoint_path_new,
                              'Tune')
            trainer.train()

            # Save all the used gin-configurable parameters
            gin_config_str = gin.operative_config_str()
            self.logger.info("GIN Configurable Parameters:\n%s", gin_config_str)


    def tune_hyper(self):
        # The approach to tune hyperparameters, the method can be 'grid', 'random', 'bayes', by default it is 'random'
        sweep_config = {
            'method': 'random',
            'name': self.model_name + '_tune',
        }
        # Set the goal of the sweep iteration
        metric = {
            'name': 'Best validation loss',
            'goal': 'minimize'
        }
        sweep_config['metric'] = metric
        # Set the hyperparameters of the parameters in the pipeline
        sweep_config['parameters'] = {}
        sweep_config['parameters'].update({
            'Trainer.epochs': {'value': 200},
            'Trainer.optimizer_type': {'values': ["Adam", "SGD", "RMSprop", "Adagrad"]},
            'Trainer.learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-1},
            'batch_size': {'distribution': 'q_uniform', 'q': 4, 'min': 4, 'max': 16},
            # 'frame_length': {'distribution': 'q_uniform', 'q': 1, 'min': 1, 'max': 5},
        })
        # The tuned hyperparameters in the CNN model
        if self.model_name == 'CNN_simple':
            sweep_config['parameters'].update({
                # 'CNN_simple.kernel_size': {'values': 2},
                'CNN_simple.num_layers': {'values': [4]},
                'CNN_simple.neuron_list': {'values': [[16, 32, 16, 2], [8, 16, 8, 2], [16, 32, 64, 2]]},
                'CNN_simple.upsampling_layer_idx': {'distribution': 'q_uniform', 'q': 1, 'min': 0, 'max': 3},
            })
        # The tuned hyperparameters in the DP model
        elif self.model_name == 'DP-Transformer':
            sweep_config['parameters'].update({
                'DPTransformerSTFT.dff': {'distribution': 'q_uniform', 'q': 8, 'min': 48, 'max': 96},
                'DPTransformerSTFT.enc_kernel_size': {'distribution': 'q_uniform', 'q': 1, 'min': 1, 'max': 5},
                'DPTransformerSTFT.dec_kernel_size': {'distribution': 'q_uniform', 'q': 1, 'min': 1, 'max': 3},
                'DPTransformerSTFT.num_dp_layers': {'distribution': 'q_uniform', 'q': 1, 'min': 1, 'max': 3},
                'DPTransformerSTFT.num_layers': {'distribution': 'q_uniform', 'q': 1, 'min': 1, 'max': 3}
            })

        pprint(sweep_config)
        # Set up the name of the sweep project
        sweep_id = wandb.sweep(sweep_config, project='MA_sweeps')
        # Set up the count of the sweep iteration, by default five combinations of the hyperparameters will be tried
        wandb.agent(sweep_id, function=self.train_func, count=5)


