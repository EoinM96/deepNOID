import numpy as np
from pydub import AudioSegment

# Supply .wav file to split, output directory for split tracks and track length below
INPUT_WAV = ''
OUTPUT_DIRECTORY = ''
TRACK_LENGTH = 30

# Calculate number of tracks that would be created for splitting into tracks of equal length TRACK_LENGTH
t1 = 0
t2 = TRACK_LENGTH*1000
input_audio = AudioSegment.from_wav(INPUT_WAV)
duration = input_audio.duration_seconds
num_tracks = np.floor(duration / TRACK_LENGTH)

# For each new track to be generated, export .wav
for i in range(num_tracks):
    audio_slice = input_audio[t1:t2]
    audio_slice.export('{}/split_{}.wav'.format(OUTPUT_DIRECTORY, i), format="wav")
    print('Exported: split_{0}'.format(i))
    t1 = t2
    t2 = t2 + t2
