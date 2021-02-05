import os
import pickle
import numpy as np


def save_npy(mfccs, save_path, save_name, file_type='pkl'):
    mfccs_path = os.path.join(save_path, save_name + '.' + file_type)
    if file_type == 'npy':
        np.save(mfccs_path, mfccs)

    if file_type == 'pkl':
        outfile = open(mfccs_path, 'wb')
        pickle.dump(mfccs, outfile)
        outfile.close()

    print('save data to:', mfccs_path)

def read_pkl(pkl_path):
    infile = open(pkl_path, 'rb')
    new_dict = pickle.load(infile)
    infile.close()

    return new_dict


if __name__ == "__main__":
    # parameters
    input_path = '/Volumes/Li_YANG/Datasets/RAVDESS/preprocess/Actor_24/01-01-08-02-02-02-24'
    audio_name = 'mfcc_sample.pkl'
    image_name = 'face_sample.pkl'

    audio_mfcc = read_pkl(os.path.join(input_path, audio_name))
    print(audio_mfcc.shape)

    image = read_pkl(os.path.join(input_path, image_name))
    print(image.shape)
