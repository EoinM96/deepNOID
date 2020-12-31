import numpy as np
import librosa
from pydub import AudioSegment


def preprocessing(wav_filepath, section_time, sample_rate=22050, num_segments=30, hop_length=512,
                  n_mfcc=13, n_fft=2048, t1=0):
    """
    Extracts MFCC data for a track

    :param wav_filepath: Full path to user .wav file
    :param section_time: Duration of section in ms (passed from prediction function in classification_preprocessing.py)
    :param sample_rate: Sample rate for tract, 22050 is standard
    :param num_segments: How many segments to split track into for MFCC evaluation
    :param hop_length: Number of samples to jump forward in track each evaluation
    :param n_mfcc: Number of MFCC coefficients
    :param n_fft: Number of fast Fourier transforms
    :param t1: Start time in track for evaluation
    :return: MFCC mapping data ready to be used in RNN-LSTM model (see classification_prediction.py)
    """

    input_audio = AudioSegment.from_wav(wav_filepath)
    t2 = t1 + (section_time*1000)
    audio_slice = input_audio[t1:t2]

    # Create temporary storage space for audio slices to write to
    # See classification_prediction.py for where this is deleted before process ends
    audio_slice.export('temp_storage/audio_slice.wav', format="wav")
    temp_audio_slice = 'temp_storage/audio_slice.wav'

    # Initialise MFCC data storage
    data = {
        "mfcc": []
    }

    signal, sr = librosa.load(temp_audio_slice, sr=sample_rate)
    duration = librosa.get_duration(signal)
    samples_per_track = sr * duration
    samples_per_segment = int(samples_per_track / num_segments)
    num_mfcc_vectors_per_segment = np.ceil(samples_per_segment / hop_length)

    # Process all segments of audio file
    for d in range(num_segments):

        # Calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # Extract MFCC's
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=n_mfcc, n_fft=n_fft,
                                    hop_length=hop_length)
        mfcc = mfcc.T

        # Store MFCC data only if it is expected length (helps deal with EoF issues)
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())

        return data
