import logging
import os
import gin
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
import wandb
from scipy import signal

from utils.data_processing import processing, rD_processing, normalisation_processing, resampling_processing, safe_log


@gin.configurable
class sd_dataset():
    def __init__(self, logger, base_dir, shift_window, frame_length, sampling_rate, tfrecord_folder, train_types, test_types, train_test_ratio, static_remove_duplicates, low_res_processing_type, high_res_processing_type, random_state=42, is_high_pass_filter=False, tqdm_status=False, disable_tqdm=True):
        self.logger = logger
        self.base_dir = base_dir
        self.total_types = list(set(train_types + test_types))
        self.shift_window = shift_window
        self.frame_length = frame_length
        self.tfrecord_folder = tfrecord_folder
        self.train_types = train_types
        self.test_types = test_types
        self.sampling_rate = sampling_rate
        self.train_test_ratio = train_test_ratio
        self.random_state = random_state
        self.static_remove_duplicates = static_remove_duplicates
        self.low_res_processing_type = low_res_processing_type
        self.high_res_processing_type = high_res_processing_type
        self.is_high_pass_filter = is_high_pass_filter
        self.tqdm_status = tqdm_status
        self.disable_tqdm = disable_tqdm

        # self.processor = Processor(self.logger)

        wandb.log({'Frame length': frame_length})


    def cube_processing(self, cube):
        # de-biasing each chirp
        avgs = np.average(cube, 1)[:, None]
        cube = cube - avgs

        # apply a high-pass filter to fast-time samples (don't know if we actually need this).
        # this gets rid of the strong reflections around r=0.
        # can be tuned with filter length (first arg) and cut-off frequency (second arg).
        # just comment this out or change the parameter values and run again to see the effect.
        if self.is_high_pass_filter:
            sos = signal.butter(2, 2e4, 'hp', fs=1.28e6, output='sos')
            cube = signal.sosfilt(sos, cube, axis=-1)
            cube = cube.astype(np.float32)

        # return cube, np.reshape(normalise_factor, (-1, 1))
        return cube


    def load_data(self):
        '''
        Load the dataset from the given path, for the dataset_2
        :return:
        '''
        self.rd_dataset = {}
        self.logger.info('------------------------Starting to load data------------------------')
        environments = [d for d in sorted(os.listdir(self.base_dir)) if d != '.DS_Store']
        for environment in tqdm(environments, desc='Loading environments', leave=self.tqdm_status, disable=self.disable_tqdm):
            self.rd_dataset['{}_downsampled'.format(environment)] = {}
            self.rd_dataset['{}_GT'.format(environment)] = {}
            # self.rd_dataset['{}_normalise'.format(environment)] = {}
            self.environment_dir = os.path.join(self.base_dir, environment)
            tempos = [i for i in sorted(os.listdir(self.environment_dir)) if i != '.DS_Store']
            for tempo in tqdm(tempos, desc='Loading tempos', leave=self.tqdm_status, disable=self.disable_tqdm):
                self.rd_dataset['{}_downsampled'.format(environment)][tempo] = []
                self.rd_dataset['{}_GT'.format(environment)][tempo] = []
                # self.rd_dataset['{}_normalise'.format(environment)][tempo] = []
                self.tempo_dir = os.path.join(self.environment_dir, tempo)
                infos = [i for i in sorted(os.listdir(self.tempo_dir)) if i != '.DS_Store']
                for info in tqdm(infos, desc='Loading infos', leave=self.tqdm_status, disable=self.disable_tqdm):
                    self.info_dir = os.path.join(self.tempo_dir, info)
                    times = [time for time in sorted(os.listdir(self.info_dir)) if time != '.DS_Store']
                    for time_id in times:
                        self.time_dir = os.path.join(self.info_dir, time_id)
                        frames = [frame for frame in sorted(os.listdir(self.time_dir)) if frame != '.DS_Store']
                        for i in range(0, len(frames), self.shift_window):
                            data_frame = []
                            for frame_length_idx in range(i, i+self.frame_length):
                                # Within the frame length, the upsampling frame is the last one
                                self.frame_dir = os.path.join(self.time_dir, frames[frame_length_idx])
                                data = np.load(self.frame_dir, allow_pickle=True)
                                # debiasing the offset in each chirp
                                data = self.cube_processing(data)
                                # Downsample the high-resolution data
                                data_downsampled = data[:data.shape[0] // self.sampling_rate, :data.shape[1] // self.sampling_rate]
                                data_frame.append(data_downsampled)
                                if frame_length_idx == i+self.frame_length-1:
                                    # Only take the last frame for the ground truth
                                    data_GT = np.array(data.reshape(1, 1, data.shape[0], data.shape[1]))
                                    data_frame = np.array(data_frame).reshape(1, self.frame_length, data_downsampled.shape[0], data_downsampled.shape[1])
                                    # Save the data according to the resolution, environment and tempo
                                    if self.rd_dataset['{}_downsampled'.format(environment)][tempo] == []:
                                        self.rd_dataset['{}_downsampled'.format(environment)][tempo] = data_frame
                                        self.rd_dataset['{}_GT'.format(environment)][tempo] = data_GT
                                        # self.rd_dataset['{}_normalise'.format(environment)][tempo] = normalized_factor
                                    else:
                                        self.rd_dataset['{}_downsampled'.format(environment)][tempo] = np.concatenate((self.rd_dataset['{}_downsampled'.format(environment)][tempo], data_frame), axis=0)
                                        self.rd_dataset['{}_GT'.format(environment)][tempo] = np.concatenate((self.rd_dataset['{}_GT'.format(environment)][tempo], data_GT), axis=0)
                                        # self.rd_dataset['{}_normalise'.format(environment)][tempo] = np.concatenate((self.rd_dataset['{}_normalise'.format(environment)][tempo], normalized_factor), axis=0)

        # Split the envs and tempos of the total types
        self.env_dict = {}
        for item in self.total_types:
            environment, tempo = item.split('&')
            if environment not in self.env_dict:
                self.env_dict[environment] = []
            self.env_dict[environment].append(tempo)

        # Logging the data shape
        for env in self.env_dict.keys():
            for tempo in self.env_dict[env]:
                self.logger.info('The size of the dataset for {}&{} case is: {}'.format(env, tempo, np.array(self.rd_dataset['{}_downsampled'.format(env)][tempo]).shape))

        self.logger.info('------------------------Finish loading data------------------------')

        return self.rd_dataset


    # def load_data(self):
    #     '''
    #     Load the dataset from the given path, for the dataset_1
    #     :return:
    #     '''
    #     self.rd_dataset = {}
    #     movingPerson_dataset = []
    #     movingPerson_GT = []
    #     movingRadar_dataset = []
    #     movingRadar_GT = []
    #     static_dataset = []
    #     static_GT = []
    #     self.logger.info('------------------------Starting to load data------------------------')
    #     dates = [d for d in sorted(os.listdir(self.base_dir)) if d != '.DS_Store']
    #     for date in tqdm(dates, desc='Process of loading dataset', leave=False):
    #         self.date_dir = os.path.join(self.base_dir, date)
    #         for type in self.total_types:
    #             self.type_dir = os.path.join(self.date_dir, type)
    #             infos = [i for i in sorted(os.listdir(self.type_dir)) if i != '.DS_Store']
    #             for info in infos:
    #                 self.info_dir = os.path.join(self.type_dir, info)
    #                 times = [time for time in sorted(os.listdir(self.info_dir)) if time != '.DS_Store']
    #                 frames = list(range(self.frame_length-1, self.total_frames))
    #
    #                 # If type is static, only keep the latest time of recording and the last frame
    #                 if self.static_remove_duplicates:
    #                     if type == 'static':
    #                         times = [x for x in times if x == max(times)]
    #                         frames = [self.total_frames-1]
    #
    #                 for time_id in times:
    #                     self.time_dir = os.path.join(self.info_dir, time_id)
    #                     for frame in frames:
    #                         data_frame = []
    #                         for frame_length_idx in range(self.frame_length-1, -1, -1):
    #                             # Within the frame length, the upsampling frame is the last one
    #                             self.frame_dir = self.time_dir + '/' + str(frame-frame_length_idx) + '.pickle'
    #                             data = np.load(self.frame_dir, allow_pickle=True)
    #                             # data_downsampled = data[::self.sampling_rate, ::self.sampling_rate]
    #                             data_downsampled = data[:data.shape[0] // self.sampling_rate, :data.shape[1] // self.sampling_rate]
    #                             # data_downsampled = range_Doppler(data_downsampled)
    #                             data_frame.append(self.cube_processing(data_downsampled))
    #                         # print(np.array(data_frame).shape)
    #                         if type == 'movingPerson':
    #                             movingPerson_dataset.append(data_frame)
    #                             movingPerson_GT.append(np.array(self.cube_processing(data)).reshape(1, data.shape[0], data.shape[1]))
    #                         elif type == 'movingRadar':
    #                             movingRadar_dataset.append(data_frame)
    #                             movingRadar_GT.append(np.array(self.cube_processing(data)).reshape(1, data.shape[0], data.shape[1]))
    #                         elif type == 'static':
    #                             static_dataset.append(data_frame)
    #                             static_GT.append(np.array(self.cube_processing(data)).reshape(1, data.shape[0], data.shape[1]))
    #
    #     if 'movingRadar' in self.total_types:
    #         self.logger.info('The size of the dataset for movingRadar case is: {}'.format(np.array(movingRadar_dataset).shape))
    #         self.logger.info('The size of the ground truth for movingRadar case is: {}'.format(np.array(movingRadar_GT).shape))
    #         self.rd_dataset['movingRadar_downsampled'] = np.array(movingRadar_dataset)
    #         self.rd_dataset['movingRadar_GT'] = np.array(movingRadar_GT)
    #     if 'movingPerson' in self.total_types:
    #         self.logger.info('The size of the dataset for movingPerson case is: {}'.format(np.array(movingPerson_dataset).shape))
    #         self.logger.info('The size of the ground truth for movingPerson case is: {}'.format(np.array(movingPerson_GT).shape))
    #         self.rd_dataset['movingPerson_downsampled'] = np.array(movingPerson_dataset)
    #         self.rd_dataset['movingPerson_GT'] = np.array(movingPerson_GT)
    #     if 'static' in self.total_types:
    #         self.logger.info('The size of the dataset for static case is: {}'.format(np.array(static_dataset).shape))
    #         self.logger.info('The size of the ground truth for static case is: {}'.format(np.array(static_GT).shape))
    #         self.rd_dataset['static_downsampled'] = np.array(static_dataset)
    #         self.rd_dataset['static_GT'] = np.array(static_GT)
    #
    #     self.logger.info('------------------------Finish loading data------------------------')
    #
    #     return self.rd_dataset


    # The following functions can be used to convert a value to a type compatible with tf.train.Example.
    def _bytes_feature(self, value):
        value = tf.io.serialize_tensor(value)
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):  # if value ist tensor
            value = value.numpy()  # get value of tensor
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        if isinstance(value, np.ndarray):
            value = value.flatten()[0]
        elif isinstance(value, (list, tuple)):
            value = value[0]
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))
        # return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a list or tuple of integers."""
        if isinstance(value, (list, tuple)):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        else:
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def write_tfrecord(self, low_resolutions, high_resolutions, path, type):

        '''
        Create TFRecord file for training or testing set
        Args:
            low_resolutions: the dataset of low resolution
            high_resolutions: the dataset of high resolution
            path: path of the folder to save tfrecord file
            type: 'train' or 'test' dataset
        '''

        with tf.io.TFRecordWriter(path) as writer:
            for low_resolution, high_resolution in zip(low_resolutions, high_resolutions):
                feature = {  # build Feature dictionary
                    'low_resolution': self._bytes_feature(low_resolution),
                    'low_resolution_shape': self._int64_feature(low_resolution.shape),
                    'high_resolution': self._bytes_feature(high_resolution),
                    'high_resolution_shape': self._int64_feature(high_resolution.shape)
                    # 'high_factor': self._float_feature(high_factor)
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))  # build Example
                writer.write(example.SerializeToString())
        self.logger.info(f"A new TFRecord file of {type} dataset is created.")


    def creating_action(self, tfrecord_suffix):

        '''
        Split the dataset into train and test dataset and call write_tfrecord function to create TFRecord files
        '''

        self.rd_dataset = self.load_data()
        # The path to save the tfrecord files
        tfrecord_train = os.path.join(self.tfrecord_folder, 'train'+tfrecord_suffix+'.tfrecord')
        tfrecord_val = os.path.join(self.tfrecord_folder, 'val'+tfrecord_suffix+'.tfrecord')
        tfrecord_test = os.path.join(self.tfrecord_folder, 'test'+tfrecord_suffix+'.tfrecord')

        if not set(self.train_types).issubset(self.total_types):
            self.logger.error("The train types are not included in the total types.")
        if not set(self.test_types).issubset(self.total_types):
            self.logger.error("The test types are not included in the total types.")

        self.logger.info('------------------------Splitted into train and test dataset------------------------')

        train_low_resolutions = []
        train_high_resolutions = []
        # train_factors = []
        test_low_resolutions = []
        test_high_resolutions = []
        # test_factors = []

        # According to the types of dataset, split the dataset into train and test
        common_types = list(set(self.train_types) & set(self.test_types))
        train_unique = [item for item in self.train_types if item not in common_types]
        test_unique = [item for item in self.test_types if item not in common_types]

        # If there are common types in train and test, split them with 80% for training and 20% for testing by default
        for common_type_idx, common_type in enumerate(common_types):
            common_environment, common_tempo = common_type.split('&')
            train_data, test_data, train_gt, test_gt = train_test_split(
                self.rd_dataset[common_environment+'_downsampled'][common_tempo],
                self.rd_dataset[common_environment+'_GT'][common_tempo],
                # self.rd_dataset[common_environment+'_normalise'][common_tempo],
                train_size=self.train_test_ratio,
                random_state=self.random_state
            )
            if common_type_idx == 0:
                train_low_resolutions = train_data
                train_high_resolutions = train_gt
                # train_factors = train_factor
                test_low_resolutions = test_data
                test_high_resolutions = test_gt
                # test_factors = test_factor
            else:
                train_low_resolutions = np.concatenate((train_low_resolutions, train_data), axis=0)
                train_high_resolutions = np.concatenate((train_high_resolutions, train_gt), axis=0)
                # train_factors = np.concatenate((train_factors, train_factor), axis=0)
                test_low_resolutions = np.concatenate((test_low_resolutions, test_data), axis=0)
                test_high_resolutions = np.concatenate((test_high_resolutions, test_gt), axis=0)
                # test_factors = np.concatenate((test_factors, test_factor), axis=0)

        self.logger.info('Dataset in common types finished.')

        self.logger.info('Train dataset is going to be split.')

        # After concatenating the dataset, there will no more the common type of dataset
        for train_type_idx, train_type in enumerate(train_unique):
            train_environment, train_tempo = train_type.split('&')
            if common_types==[] and train_type_idx == 0:
                train_low_resolutions = self.rd_dataset[train_environment+'_downsampled'][train_tempo]
                train_high_resolutions = self.rd_dataset[train_environment+'_GT'][train_tempo]
                # train_factors = self.rd_dataset[train_environment+'_normalise'][train_tempo]
            else:
                train_low_resolutions = np.concatenate((train_low_resolutions, self.rd_dataset[train_environment+'_downsampled'][train_tempo]), axis=0)
                train_high_resolutions = np.concatenate((train_high_resolutions, self.rd_dataset[train_environment+'_GT'][train_tempo]), axis=0)
                # train_factors = np.concatenate((train_factors, self.rd_dataset[train_environment+'_normalise'][train_tempo]), axis=0)

        self.logger.info(f"Train low resolutions shape: {train_low_resolutions.shape}")
        self.logger.info(f"Train high resolutions shape: {train_high_resolutions.shape}")
        # self.logger.info(f"Train factors shape: {train_factors.shape}")

        self.logger.info('Validation dataset is going to be split.')

        # Split the validation dataset from train dataset
        train_low_resolutions, val_low_resolutions, train_high_resolutions, val_high_resolutions = train_test_split(
            train_low_resolutions,
            train_high_resolutions,
            # train_factors,
            train_size=self.train_test_ratio,
            random_state=self.random_state
        )

        self.logger.info(f"Validation low resolutions shape: {val_low_resolutions.shape}")
        self.logger.info(f"Validation high resolutions shape: {val_high_resolutions.shape}")
        # self.logger.info(f"Validation factors shape: {val_factors.shape}")

        self.write_tfrecord(train_low_resolutions, train_high_resolutions, tfrecord_train, 'train')
        self.write_tfrecord(val_low_resolutions, val_high_resolutions, tfrecord_val, 'val')

        del train_low_resolutions, train_high_resolutions, val_low_resolutions, val_high_resolutions

        self.logger.info('Test dataset is going to be split.')

        for test_type_idx, test_type in enumerate(test_unique):
            test_environment, test_tempo = test_type.split('&')
            if common_types==[]  and test_type_idx == 0:
                test_low_resolutions = self.rd_dataset[test_environment+'_downsampled'][test_tempo]
                test_high_resolutions = self.rd_dataset[test_environment+'_GT'][test_tempo]
                # test_factors = self.rd_dataset[test_environment+'_normalise'][test_tempo]
            else:
                test_low_resolutions = np.concatenate((test_low_resolutions, self.rd_dataset[test_environment+'_downsampled'][test_tempo]), axis=0)
                test_high_resolutions = np.concatenate((test_high_resolutions, self.rd_dataset[test_environment+'_GT'][test_tempo]), axis=0)
                # test_factors = np.concatenate((test_factors, self.rd_dataset[test_environment+'_normalise'][test_tempo]), axis=0)

        self.logger.info(f"Test low resolutions shape: {test_low_resolutions.shape}")
        self.logger.info(f"Test high resolutions shape: {test_high_resolutions.shape}")
        # self.logger.info(f"Test factors shape: {test_factors.shape}")

        self.write_tfrecord(test_low_resolutions, test_high_resolutions, tfrecord_test, 'test')

        del test_low_resolutions, test_high_resolutions, self.rd_dataset

        self.logger.info('------------------------Finished creating TFRecord files------------------------')


    @gin.configurable
    def prepare(self, ds_train, ds_val, ds_test, buffer_size, shuffle_seed, batch_size, caching=False):

        '''
        Prepare the dataset, such as batch, augment, prefetch and so on
        Args:
            ds_train: train set
            ds_test: test set
            buffer_size: size of shuffle buffer
            batch_size: size of dataset batch
            caching: whether caching or not
        Return:
            ds_train: train set after preparing
            ds_test: test set after preparing
        '''

        self.logger.info('Preparing the datasets now...')
        wandb.log({'Batch size': batch_size})
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        # Prepare training dataset
        # ds_train = ds_train.map(lambda low_res, high_res: (resampling_processing(low_res), high_res))
        ds_train = ds_train.map(lambda low_res, high_res: (rD_processing(low_res), rD_processing(high_res)))
        # Find the global maximum value in the training set
        global_train_max = ds_train.reduce(0.0 ,lambda current_max, elem: tf.maximum(current_max, tf.reduce_max(abs(tf.math.log(safe_log(tf.math.abs(elem[0])))/tf.math.log(tf.constant(10.0, dtype=tf.float32))))))
        # Find the global minimum value in the training set
        global_train_min = ds_train.reduce(0.0, lambda current_min, elem: tf.minimum(current_min, tf.reduce_min(tf.math.log(safe_log(tf.math.abs(elem[0])))/tf.math.log(tf.constant(10.0, dtype=tf.float32)))))
        # Calculate the sum of all the values and the total number of samples in training set for obtaining the mean value in the training set
        total_sum, total_count = ds_train.reduce(
            (0.0, 0),
            lambda state, elem: (
                state[0] + tf.reduce_sum(
                    tf.math.log(safe_log(tf.abs(elem[0]))) / tf.math.log(tf.constant(10.0, dtype=tf.float32))),
                state[1] + tf.size(elem[0])
            )
        )
        #
        global_train_mean = total_sum / tf.cast(total_count, tf.float32)
        # global_train_max = ds_train.reduce(0.0 ,lambda current_max, elem: tf.maximum(current_max, tf.reduce_max(tf.abs(elem[0]))))
        # ds_train = ds_train.map(lambda low_res, high_res: (normalisation_processing(low_res), normalisation_processing(high_res)))
        # ds_train = ds_train.map(lambda low_res, high_res: (low_res[0], low_res[1], high_res[0], high_res[1]))

        # ds_train = ds_train.map(lambda low_res, high_res: (processing(low_res, 'low_res', self.low_res_processing_type, self.logger), processing(high_res, 'high_res', self.high_res_processing_type, self.logger)))
        if caching:
            ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(buffer_size, seed=shuffle_seed)
        ds_train = ds_train.batch(batch_size, drop_remainder=True)
        # ds_train = ds_train.repeat(5)
        ds_train = ds_train.prefetch(AUTOTUNE)
        self.logger.info('Train set is prepared')

        # Prepare validation dataset
        # ds_val = ds_val.map(lambda low_res, high_res: (resampling_processing(low_res), high_res))
        ds_val = ds_val.map(lambda low_res, high_res: (rD_processing(low_res), rD_processing(high_res)))
        global_val_max = ds_val.reduce(0.0, lambda current_max, elem: tf.maximum(current_max, tf.reduce_max(abs(tf.math.log(safe_log(tf.math.abs(elem[0])))/tf.math.log(tf.constant(10.0, dtype=tf.float32))))))
        global_val_min = ds_val.reduce(0.0, lambda current_min, elem: tf.minimum(current_min, tf.reduce_min(tf.math.log(safe_log(tf.math.abs(elem[0])))/tf.math.log(tf.constant(10.0, dtype=tf.float32)))))
        total_val_sum, total_val_count = ds_val.reduce(
            (0.0, 0),
            lambda state, elem: (
                state[0] + tf.reduce_sum(
                    tf.math.log(safe_log(tf.abs(elem[0]))) / tf.math.log(tf.constant(10.0, dtype=tf.float32))),
                state[1] + tf.size(elem[0])
            )
        )
        global_val_mean = total_val_sum / tf.cast(total_val_count, tf.float32)
        # global_val_max = ds_val.reduce(0.0, lambda current_max, elem: tf.maximum(current_max, tf.reduce_max(tf.abs(elem[0]))))

        # ds_val = ds_val.map(lambda low_res, high_res: (processing(low_res, 'low_res', self.low_res_processing_type, self.logger), processing(high_res, 'high_res', self.high_res_processing_type, self.logger)))
        if caching:
            ds_val = ds_val.cache()
        ds_val = ds_val.batch(batch_size, drop_remainder=True)
        ds_val = ds_val.prefetch(AUTOTUNE)
        self.logger.info('Validation set is prepared')

        # Prepare test dataset
        # ds_test = ds_test.map(lambda low_res, high_res: (resampling_processing(low_res), high_res))
        ds_test = ds_test.map(lambda low_res, high_res: (rD_processing(low_res), rD_processing(high_res)))
        global_test_max = ds_test.reduce(0.0, lambda current_max, elem: tf.maximum(current_max, tf.reduce_max(abs(tf.math.log(safe_log(tf.math.abs(elem[0])))/tf.math.log(tf.constant(10.0, dtype=tf.float32))))))
        global_test_min = ds_test.reduce(0.0, lambda current_min, elem: tf.minimum(current_min, tf.reduce_min(tf.math.log(safe_log(tf.math.abs(elem[0])))/tf.math.log(tf.constant(10.0, dtype=tf.float32)))))
        total_test_sum, total_test_count = ds_test.reduce(
            (0.0, 0),
            lambda state, elem: (
                state[0] + tf.reduce_sum(
                    tf.math.log(safe_log(tf.abs(elem[0]))) / tf.math.log(tf.constant(10.0, dtype=tf.float32))),
                state[1] + tf.size(elem[0])
            )
        )
        global_test_mean = total_test_sum / tf.cast(total_test_count, tf.float32)
        # global_test_max = ds_test.reduce(0.0, lambda current_max, elem: tf.maximum(current_max, tf.reduce_max(tf.abs(elem[0]))))

        # ds_test = ds_test.map(lambda low_res, high_res: (processing(low_res, 'low_res', self.low_res_processing_type, self.logger), processing(high_res, 'high_res', self.high_res_processing_type, self.logger)))
        ds_test = ds_test.batch(batch_size, drop_remainder=True)
        if caching:
            ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(AUTOTUNE)
        self.logger.info('Test set is prepared')

        self.logger.info('------------------------Finish preparing the datasets------------------------')

        return ds_train, ds_val, ds_test, global_train_max, global_train_mean, global_train_min, global_val_max, global_val_mean, global_val_min, global_test_max, global_test_mean, global_test_min


    def load_tfrecord(self, tfrecord_suffix):

        '''
        Load TFRecord file
        '''

        self.logger.info('------------------------Loading TFRecord files------------------------')
        self.tfrecord_train = os.path.join(self.tfrecord_folder, 'train'+tfrecord_suffix+'.tfrecord')
        self.tfrecord_val = os.path.join(self.tfrecord_folder, 'val'+tfrecord_suffix+'.tfrecord')
        self.tfrecord_test = os.path.join(self.tfrecord_folder, 'test'+tfrecord_suffix+'.tfrecord')

        ds_info = {
            'low_resolution': tf.io.FixedLenFeature([], tf.string),
            'low_resolution_shape': tf.io.FixedLenFeature([3], tf.int64),
            'high_resolution': tf.io.FixedLenFeature([], tf.string),
            'high_resolution_shape': tf.io.FixedLenFeature([3], tf.int64)
        }

        # Parse the dataset
        def _parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, ds_info)

            low_resolution_raw = feature_dict['low_resolution']
            low_resolution_shape = feature_dict['low_resolution_shape']
            low_resolution = tf.io.parse_tensor(low_resolution_raw, tf.float32)
            feature_dict['low_resolution'] = tf.reshape(low_resolution, low_resolution_shape)

            high_resolution_raw = feature_dict['high_resolution']
            high_resolution_shape = feature_dict['high_resolution_shape']
            high_resolution = tf.io.parse_tensor(high_resolution_raw, tf.float32)
            feature_dict['high_resolution'] = tf.reshape(high_resolution, high_resolution_shape)

            return feature_dict['low_resolution'], feature_dict['high_resolution']

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        # Import the train set from TFRecord file
        ds_train = tf.data.TFRecordDataset(self.tfrecord_train)
        ds_train = ds_train.map(_parse_example, num_parallel_calls=AUTOTUNE)

        # # Check the shape of the low resolution and high resolution data
        # for idx, (low_res, high_res) in enumerate(ds_train.take(1)):
        #     self.low_res = low_res
        #     self.high_res = high_res
        #     print(f"Low resolution shape: {low_res.shape}")
        #
        # # Check if the loaded dataset is correct
        # # if tf.reduce_all(tf.math.abs(self.low_res_origin - self.low_res) < 1e-6) and tf.reduce_all(tf.math.abs(self.high_res_origin - self.high_res) < 1e-6):
        # if tf.reduce_all(tf.equal(self.low_res_origin, self.low_res)) and tf.reduce_all(tf.equal(self.high_res_origin, self.high_res)):
        #     self.logger.info('The loaded dataset is correct.')

        # Import the val set from TFRecord file
        ds_val = tf.data.TFRecordDataset(self.tfrecord_val)
        ds_val = ds_val.map(_parse_example, num_parallel_calls=AUTOTUNE)

        # Import the test set from TFRecord file
        ds_test = tf.data.TFRecordDataset(self.tfrecord_test)
        ds_test = ds_test.map(_parse_example, num_parallel_calls=AUTOTUNE)

        return self.prepare(ds_train, ds_val, ds_test)


