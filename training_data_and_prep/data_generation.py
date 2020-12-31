import numpy as np
import os
import librosa
import json

DATASET_PATH = "raw_training_data"
JSON_PATH = "raw_training_data/usable_data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=30):
    """
    Saves MFFC's from data in DATASET_PATH and exports them to JSON_PATH

    :param dataset_path: Path to users training data
    :param json_path: Path to location where JSON file is to write to (file name and extension included)
    :param num_mfcc: Number of MFCC coefficients
    :param n_fft: Number of fast Fourier transforms
    :param hop_length: Number of samples to jump forward in track each evaluation
    :param num_segments: How many segments to split track into for MFCC evaluation
    :return: JSON file of data ready to use in RNN-LSTM model
    """

    # Dictionary to store mapping, labels, and MFCC's
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    # Loop through all genre folders and tracks within them
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Ensure we aren't at top level path
        if dirpath is not dataset_path:

            # Save genre label/subfolder name in mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # For all tracks in genre subfolder
            for f in filenames:

                # Load track
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                samples_per_track = SAMPLE_RATE * TRACK_DURATION
                samples_per_segment = int(samples_per_track / num_segments)
                num_mfcc_vectors_per_segment = np.ceil(samples_per_segment / hop_length)

                # Process each segment of track
                for d in range(num_segments):

                    # Calc start/end time for segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # MFCC extraction
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # Store MFCC data only if it is expected length (helps deal with EoF issues)
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, d + 1))

    # Save MFCC's to JSON file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)
