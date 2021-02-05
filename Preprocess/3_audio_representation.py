import librosa.display  # for waveplots, spectograms, etc
import tensorflow as tf
from comman_tools import *

if __name__ == '__main__':
    INPUT_FILES = '/Volumes/Li_YANG/Datasets/RAVDESS/preprocess'

    Audio_rep = []
    for root, dires, files in os.walk(INPUT_FILES):
        if files and 'audio.wav' in files:
            audio_path = os.path.join(root, 'audio.wav')
            print('load audio from: \n', audio_path)
            # Extract the audio from the video
            audio, sample_rate = librosa.load(audio_path, sr=None)  # 1.audio time series, 2.sampling rate
            # Extract 24 Mel Frequency Cepstral Coefficients from the audio
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=24)
            # Calculate the mean number of (spectral) frames in the dataset
            ave_frams = 188
            # Standardize the MFCCs sample-wise
            std_audio = (mfccs - np.mean(mfccs)) / np.std(mfccs)
            # Use pre-padding to unify the length of the samples
            audio_array = tf.keras.preprocessing.sequence.pad_sequences(std_audio.tolist(), maxlen=ave_frams)

            # save to a tensor with shape (N,M,1) = (24,188,1)
            audio_array = np.expand_dims(audio_array, axis=-1)
            audio_tensor = tf.convert_to_tensor(audio_array)
            save_npy(audio_tensor, root, 'mfcc_sample')
