import numpy as np
import librosa
from pydub import AudioSegment


def preprocessing(wav_filepath, section_time, sample_rate=22050, num_segments=30, hop_length=512,
                  n_mfcc=13, n_fft=2048, t1=0):
    input_audio = AudioSegment.from_wav(wav_filepath)
    t2 = t1 + (section_time*1000)
    audio_slice = input_audio[t1:t2]
    audio_slice.export('temp_storage/audio_slice.wav', format="wav")
    temp_audio_slice = 'temp_storage/audio_slice.wav'

    data = {
        "mfcc": []
    }

    signal, sr = librosa.load(temp_audio_slice, sr=sample_rate)
    duration = librosa.get_duration(signal)
    samples_per_track = sr * duration
    samples_per_segment = int(samples_per_track / num_segments)
    num_mfcc_vectors_per_segment = np.ceil(samples_per_segment / hop_length)

    # process all segments of audio file
    for d in range(num_segments):

        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=n_mfcc, n_fft=n_fft,
                                    hop_length=hop_length)
        mfcc = mfcc.T

        # store only mfcc feature with expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())

        return data
