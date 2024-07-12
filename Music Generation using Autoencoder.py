#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
import numpy as np
import os
import pickle

class Autoencoder:
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape = input_shape  # [w, h, c]
        self.conv_filters = conv_filters  # []
        self.conv_kernels = conv_kernels  # []
        self.conv_strides = conv_strides  # []
        self.latent_space_dim = latent_space_dim  # int
        self._num_conv_layers = len(conv_filters)
        self.encoder = None
        self.decoder = None
        self.model = None
        self._shape_before_bottleneck = None
        self._model_input = None
        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return layers.Input(shape=self.input_shape, name='encoder_input')

    def _add_conv_layers(self, encoder_input):
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        layer_number = layer_index + 1
        conv_layer = layers.Conv2D(filters=self.conv_filters[layer_index],
                                   kernel_size=self.conv_kernels[layer_index],
                                   strides=self.conv_strides[layer_index],
                                   padding="same",
                                   name=f"encoder_conv_layer_{layer_number}")
        x = conv_layer(x)
        x = layers.ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = layers.BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, conv_layers):
        self._shape_before_bottleneck = K.int_shape(conv_layers)[1:]  # [batch_size, w, h, c] -> ignore batch_size
        x = layers.Flatten(name="bottleneck")(conv_layers)
        x = layers.Dense(self.latent_space_dim, name="encoder_output")(x)
        return x

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return layers.Input(shape=(self.latent_space_dim,), name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = layers.Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        reshape_layer = layers.Reshape(target_shape=self._shape_before_bottleneck, name="decoder_reshape")(dense_layer)
        return reshape_layer

    def _add_conv_transpose_layers(self, reshape_layer):
        x = reshape_layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_number = self._num_conv_layers - layer_index
        conv_transpose_layer = layers.Conv2DTranspose(filters=self.conv_filters[layer_index],
                                                      kernel_size=self.conv_kernels[layer_index],
                                                      strides=self.conv_strides[layer_index],
                                                      padding="same",
                                                      name=f"decoder_conv_transpose_layer_{layer_number}")
        x = conv_transpose_layer(x)
        x = layers.ReLU(name=f"decoder_relu_{layer_number}")(x)
        x = layers.BatchNormalization(name=f"decoder_bn_{layer_number}")(x)
        return x

    def _add_decoder_output(self, conv_transpose_layers):
        conv_transpose_layer = layers.Conv2DTranspose(filters=1,
                                                      kernel_size=self.conv_kernels[0],
                                                      strides=self.conv_strides[0],
                                                      padding="same",
                                                      name=f"decoder_conv_transpose_layer_{self._num_conv_layers}")
        x = conv_transpose_layer(conv_transpose_layers)
        output_layer = layers.Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def compile(self, learning_rate=0.001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        mse_loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train(self, X_train, batch_size, epochs):
        self.model.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, shuffle=True)

    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations


if __name__ == "__main__":
    autoencoder = Autoencoder(input_shape=(256, 752, 1),  # Adjusted input shape for 8-second samples
                              conv_filters=[32, 64, 128, 256],
                              conv_kernels=[3, 3, 3, 3],
                              conv_strides=[1, 2, 2, 2],  # Adjusted strides for larger input
                              latent_space_dim=128)  # Adjusted latent space dimension
    autoencoder.summary()


# In[2]:


import numpy as np
import librosa
import os
import pickle

class Loader:
    """
    Loader is responsible for loading an audio file.
    """
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0] # returns a tuple: (signal, sample_rate)
        return signal


class Padder:
    """
    Padder is responsible for applying padding to an array.
    """
    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        """
        eg: [1,2,3] with 2 items -> [0,0,1,2,3]
        """
        padded_array = np.pad(array, (num_missing_items, 0), mode=self.mode) # insert/append num_missing_items at the beginning of the array
        return padded_array

    def right_pad(self, array, num_missing_items):
        """
        eg: [1,2,3] with 2 items -> [1,2,3,0,0]
        """
        padded_array = np.pad(array, (0, num_missing_items),
                              mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    """
    Extracts Log Spectrograms (in dB) from a time-series signal.
    """
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        # short-time fourier transform
        stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length)[:-1]  # (1 + (frame_size / 2), num_frames)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


class MinMaxNormaliser:
    """
    Applies min-max normalization to an array. Using range [0,1].
    """
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def normalise(self, array):
        normalised_array = (array - array.min()) / (array.max() - array.min()) # (x - xmin) / (xmax - xmin)
        normalised_array = normalised_array * (self.max_val - self.min_val) + self.min_val
        return normalised_array

    def denormalise(self, normalised_array, original_min_of_array, original_max_of_array):
        array = (normalised_array - self.min_val) / (self.max_val - self.min_val)
        array = array * (original_max_of_array - original_min_of_array) + original_min_of_array
        return array


class Saver:
    """
    Responsible for saving the features, and the min max values which will further be used during reconstruction.
    """
    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir
        self._ensure_dir_exists(self.feature_save_dir)
        self._ensure_dir_exists(self.min_max_values_save_dir)

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        return save_path

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1] # returns [head, tail]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(min_max_values, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(min_max_values, f)

    @staticmethod
    def _ensure_dir_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


class PreprocessingPipeline:
    """
    Preprocesses the audio files in a directory, applying the following steps to each file:
        1. Load the file
        2. Pad the signal (if necessary)
        3. Extracting log spectrograms from the signal
        4. Normalize spectrogram
        5. Save the normalized spectrogram

    Storing the min max values for all the log spectrograms for reconstructing the signal.
    """
    def __init__(self, loader, padder, extractor, normaliser, saver):
        self.loader = loader
        self.padder = padder
        self.extractor = extractor
        self.normaliser = normaliser
        self.saver = saver
        self.min_max_values = {} # {save_path: {"min": min_val, "max": max_val}}
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def preprocess(self, audio_file_dir):
        for root, _, files in os.walk(audio_file_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    self._process_file(file_path)
                    print(f"Processed file: {file_path}")
                except Exception as e:
                    print(f"Could not process file {file_path}: {e}")
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_required(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        normalised_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(normalised_feature, file_path)
        self._store_min_max_values(save_path, feature.min(), feature.max())

    def _is_padding_required(self, signal):
        return len(signal) < self._num_expected_samples

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    def _store_min_max_values(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }

if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 5 # in seconds
    SAMPLE_RATE = 22050
    MONO = True

    # Relative paths
    SPECTOGRAMS_SAVE_DIR = os.path.join(os.getcwd(), "saved_data", "spectrograms")
    MIN_MAX_VALUES_SAVE_DIR = os.path.join(os.getcwd(), "saved_data", "min_max_values")
    AUDIO_FILES_DIR = r"C:\Users\jmdgo\Downloads\archive (9)\Data\genres_original"

    # instantiate all objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(SPECTOGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    # preprocessing pipeline
    preprocessing_pipeline = PreprocessingPipeline(loader, padder, log_spectrogram_extractor, min_max_normaliser, saver)

    preprocessing_pipeline.preprocess(AUDIO_FILES_DIR)


# In[3]:


import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
import numpy as np
import os
import pickle

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 5
SPECTOGRAM_PATH = r"C:\Users\jmdgo\saved_data\spectrograms"
TARGET_SHAPE = (256, 752)


def load_fsdd(spectrograms_path, target_shape):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension
            if spectrogram.shape[:2] != target_shape:
                spectrogram = tf.image.resize(spectrogram, target_shape).numpy()
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = Autoencoder(
        input_shape=(256, 752, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train = load_fsdd(SPECTOGRAM_PATH, TARGET_SHAPE)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")


# In[4]:


class SoundGenerator:
    """
    SoundGenerator is responsible for generating audios from spectograms
    """
    def __init__(self, ae, hop_length):
        self.ae = ae
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormaliser(0, 1)

    def generate(self, spectograms, min_max_values):
        generated_spectograms, latent_representations = self.ae.reconstruct(spectograms)
        signals = self.convert_spectograms_to_audio(generated_spectograms, min_max_values)
        return signals, latent_representations

    def convert_spectograms_to_audio(self, spectograms, min_max_values):
        signals = []
        for spectogram, min_max_value in zip(spectograms, min_max_values):
            # reshape the log spectogram
            log_spectogram = spectogram[:,:,0]
            # apply denormalisation
            denormalised_log_spec = self._min_max_normaliser.denormalise(log_spectogram,
                                                                          min_max_value["min"], min_max_value["max"])
            # log spectogram -> spectogram
            spec = librosa.db_to_amplitude(denormalised_log_spec)
            # apply Griffin-Lim algorithm (used inverse short-time fourier transform)
            signal = librosa.istft(spec, hop_length=self.hop_length)
            # append signal to signals list
            signals.append(signal)

        return signals


# In[6]:


import os
import pickle
import numpy as np
import soundfile as sf

HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = r"C:\Users\jmdgo\Downloads\archive (9)\Data\genres_original"
SAVE_DIR_GENERATED = r"C:\Users\jmdgo\saved_data\samples\generated"
MIN_MAX_VALUES_PATH = r"C:\Users\jmdgo\saved_data\min_max_values\min_max_values.pkl"

def load_fsdd(spectrograms_path, target_shape=(256, 752)):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            spectrogram = adjust_spectrogram_shape(spectrogram, target_shape)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]
    return x_train, file_paths

def adjust_spectrogram_shape(spectrogram, target_shape):
    target_bins, target_frames = target_shape
    bins, frames = spectrogram.shape

    if bins < target_bins:
        pad_bins = target_bins - bins
        spectrogram = np.pad(spectrogram, ((0, pad_bins), (0, 0)), mode='constant')
    elif bins > target_bins:
        spectrogram = spectrogram[:target_bins, :]

    if frames < target_frames:
        pad_frames = target_frames - frames
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_frames)), mode='constant')
    elif frames > target_frames:
        spectrogram = spectrogram[:, :target_frames]

    return spectrogram

def select_spectrograms(spectrograms, file_paths, min_max_values, num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrograms = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in file_paths]
    print(file_paths)
    print(sampled_min_max_values)
    return sampled_spectrograms, sampled_min_max_values

def save_signals(signals, save_dir, sample_rate=22050):
    os.makedirs(save_dir, exist_ok=True)
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)

if __name__ == "__main__":
    # initialise sound generator
    ae = Autoencoder.load("model")
    sound_generator = SoundGenerator(ae, HOP_LENGTH)

    # load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_fsdd(SPECTOGRAM_PATH)

    # sample spectrograms + min max values
    sampled_specs, sampled_min_max_values = select_spectrograms(specs, file_paths, min_max_values, 5)

    # generate audio for sampled spectrograms
    signals, _ = sound_generator.generate(sampled_specs, sampled_min_max_values)

    # convert spectrogram samples to audio
    original_signals = sound_generator.convert_spectograms_to_audio(sampled_specs, sampled_min_max_values)

    # save audio signals
    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)


# In[10]:


autoencoder.save(r"C:\Users\jmdgo\saved_data")

