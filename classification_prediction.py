from audio_preprocessing import preprocessing

import os
import numpy as np
import tensorflow.keras as tf
from pydub import AudioSegment


def prediction(wav_filepath, num_sections=25):
    """
    Predicts classification of user provided .wav file using cloned Tensorflow model

    :param wav_filepath: Full path to user .wav file
    :param num_sections: Number of sections track is split into for individual evaluation (higher=more accurate)
    :return: Printout if classification result
    """

    input_audio = AudioSegment.from_wav(wav_filepath)
    duration = input_audio.duration_seconds
    section_time = np.floor(duration / num_sections)  # Length of each track subsection

    noid_vec = []  # Initialise empty list to store classifications of each section

    model = tf.models.load_model('models')

    # Loop through each section of track, extract MFCC's, predict classification,
    # append to list of classifications for each section
    for i in range(num_sections):
        data = preprocessing(wav_filepath, t1=i*section_time*1000, section_time=section_time)  # Extract MFCC's

        x = np.array(data['mfcc'])  # Convert to numpy array

        # Predict and append to noid_vec
        pred = model.predict(x)
        pred = np.argmax(pred, axis=1)
        noid_vec.append(int(pred))

    # Remove track section created in preprocessing function call
    os.remove('temp_storage/audio_slice.wav')

    print('\n{} is...\n'.format(wav_filepath.split('/')[-1]))

    # If most common classification is 0 (ie noided),
    if max(set(noid_vec), key=noid_vec.count) == 0:
        print('███╗   ██╗ ██████╗ ██╗██████╗ ███████╗██████╗\n' +
              '████╗  ██║██╔═══██╗██║██╔══██╗██╔════╝██╔══██╗\n' +
              '██╔██╗ ██║██║   ██║██║██║  ██║█████╗  ██║  ██║\n' +
              '██║╚██╗██║██║   ██║██║██║  ██║██╔══╝  ██║  ██║\n' +
              '██║ ╚████║╚██████╔╝██║██████╔╝███████╗██████╔╝\n' +
              '╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═════╝ ╚══════╝╚═════╝')
    else:
        print('not noided >:(')
