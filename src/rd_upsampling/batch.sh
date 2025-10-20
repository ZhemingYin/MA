#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=1513
#SBATCH --output=job_name%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --time=5-00:00:00
#SBATCH --gpus=geforce_rtx_2080_ti:2
#SBATCH --qos=batch

# Activate everything you need
module load cuda/11.2
pyenv activate d1513
# Run your python code
python3.6 main.py \
  --gin_file=utils/config.gin \
  --gin_bind="main.type = 'Train'" \
  --gin_bind="Trainer.epochs = 500" \
  --gin_bind="main.model_name = 'DP-Transformer'" \
  --gin_bind="Trainer.learning_rate = 0.0009137" \
  --gin_bind="Trainer.loss_type = 'lsd&frequency_domain'" \
  --gin_bind="processing_method = 'padding&shuffle&ap/ph&with_log10&abs_normalization(0,1)&no_angle_normalization'" \
  --gin_bind="batch_size = 16" \
  --gin_bind="SwinIR.dp_depths = [1]" \
  --gin_bind="SwinIR.dp_num_layer = 1" \
  --gin_bind="SwinIR.swin_depths = [1,1]" \
  --gin_bind="SwinIR.patch_size = (2,2)" \
  --gin_bind="SwinIR.window_size = [5,4]" \
  --gin_bind="SwinIR.embed_dim_swin = 64" \
  --gin_bind="SwinIR.num_mlp = 256" \
  --gin_bind="DPTransformerSTFT_test.num_dp_layers = 1" \
  --gin_bind="DPTransformerSTFT_test.num_layers = 1" \
  --gin_bind="DPTransformerSTFT.num_dp_layers = 1" \
  --gin_bind="DPTransformerSTFT.num_layers = 1" \
  --gin_bind="DPTransformerSTFT.enc_kernel_size = 3" \
  --gin_bind="DPTransformerSTFT.dec_kernel_size = 1" \
  --gin_bind="DPTransformerSTFT.dff = 64" \
  --gin_bind="CNN_simple.neuron_list = [16, 32, 64, 128, 128, 64, 32, 16]" \
  --gin_bind="CNN_simple.upsampling_layer_idx = 3" \
  --gin_bind="UNet_simple.downsample_neuron_list = [16]" \
  --gin_bind="UNet_simple.upsample_neuron_list = [32, 16]" \
  --gin_bind="UNet_concat.large_model = False" \
  --gin_bind="ConditionalGAN.generator_name = 'DP-Transformer'" \
  --gin_bind="ConditionalGAN.discriminator_scale = 'small'" \
  --gin_bind="sampling_rate = 2" \
  --gin_bind="main.tfrecord_suffix = '_factor2'" \
  --gin_bind="Trainer.pretraining = True" \
  --gin_bind="Trainer.layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4']" \
  --gin_bind="Trainer.lambda_lsd = 1.0" \
  --gin_bind="Trainer.lambda_vgg = 0.5" \
  --gin_bind="Trainer.lambda_disc = 0.0" \
  --gin_bind="generator_loss.LAMBDA_l1_loss = 1.0" \
  --gin_bind="generator_loss.LAMBDA_perceptual_loss = 0.5" \
  --gin_bind="generator_loss.LAMBDA_gan_loss = 1e-2" \
  --gin_bind="DPTransformerSTFT_test.conditions = [False, False, False, False, False, False, False, True, True, False, False, False]"

