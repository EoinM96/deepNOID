import numpy as np
from pydub import AudioSegment

INPUT_WAV = ''
OUTPUT_DIRECTORY = ''

t1 = 0
t2 = 30000

input_audio = AudioSegment.from_wav(INPUT_WAV)
duration = input_audio.duration_seconds
print(duration)
num_tracks = np.floor(duration / 5)
print(num_tracks)

for i in range(num_tracks):
    print(i)
    audio_slice = input_audio[t1:t2]
    audio_slice.export('{}/split_{}.wav'.format(OUTPUT_DIRECTORY, i), format="wav")
    print('Exported: split_{0}'.format(i))
    t1 = t2
    t2 = t2 + 5000

print('Complete')
