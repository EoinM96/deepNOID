import os
from pydub import AudioSegment

# Supply directory containing .wav files to be joined into one in the empty quotes below
WAV_DIR = ''


indv_tracks = []  # Initialise list to store individual tracks
cmbd_tracks = 0  # Initialise value to combine tracks on

# Add each track in directory to indv_tracks
for file in os.listdir(WAV_DIR):
    if file.endswith(".wav"):
        wav_path = os.path.join(WAV_DIR, file)
        indv_tracks.append(AudioSegment.from_wav(wav_path))

# Combine all tracks into one
for i in range(len(indv_tracks)):
    cmbd_tracks = cmbd_tracks + indv_tracks[i]

# Export combined tracks to orginal directory
cmbd_tracks.export(WAV_DIR, format='wav')
