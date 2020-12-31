from audio_preprocessing import preprocessing

import os
import numpy as np
import tensorflow.keras as tf
from pydub import AudioSegment


def prediction(wav_filepath, num_sections=25):
    input_audio = AudioSegment.from_wav(wav_filepath)
    duration = input_audio.duration_seconds
    section_time = np.floor(duration / num_sections)

    noid_vec = []

    model = tf.models.load_model('models')

    for i in range(num_sections):
        data = preprocessing(wav_filepath, t1=i*section_time*1000, section_time=section_time)

        x = np.array(data['mfcc'])

        pred = model.predict(x)
        pred = np.argmax(pred, axis=1)
        noid_vec.append(int(pred))

    os.remove('temp_storage/audio_slice.wav')

    print('\n{} is...\n'.format(wav_filepath.split('/')[-1]))

    if max(set(noid_vec), key=noid_vec.count) == 0:
        print('███╗   ██╗ ██████╗ ██╗██████╗ ███████╗██████╗\n' +
              '████╗  ██║██╔═══██╗██║██╔══██╗██╔════╝██╔══██╗\n' +
              '██╔██╗ ██║██║   ██║██║██║  ██║█████╗  ██║  ██║\n' +
              '██║╚██╗██║██║   ██║██║██║  ██║██╔══╝  ██║  ██║\n' +
              '██║ ╚████║╚██████╔╝██║██████╔╝███████╗██████╔╝\n' +
              '╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═════╝ ╚══════╝╚═════╝')
    else:
        print('not noided >:(')
