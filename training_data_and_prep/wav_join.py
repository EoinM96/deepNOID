import os
from pydub import AudioSegment

WAV_DIR = ''
INDV_TRACKS = []
CMBD_TRACKS = 0


for file in os.listdir(WAV_DIR):
    if file.endswith(".wav"):
        wav_path = os.path.join("/mydir", file)
        INDV_TRACKS.append(AudioSegment.from_wav(wav_path))

for i in range(len(INDV_TRACKS)):
    CMBD_TRACKS = CMBD_TRACKS + INDV_TRACKS[i]

CMBD_TRACKS.export(WAV_DIR, format='wav')
