# deepNOID

deepNOID, the binary music genre classifier to determine if what you're listening to really is NOIDED.

This project was started to help me learn how to use GitHub in conjunction with PyCharm and further, how to utilise Tensorflow2/Keras. 
My background is in Mathematics and I've been teaching myself the foundations of Deep Learning for the past month. Wanting to apply what've I've beening learing into a model, deepNOID was born. 

At it's core, this is a RNN-LSTM based music genre classification algorithm trained with two genre labels, noided and not noided (ie does this sound like a Death Grips track or not).


## Getting Started

To use deepNOID,

- Clone the repo
- Ensure dependencies are installed by running `pip install -r requirements.txt`
- Open `deepNOID.py` and insert path to .wav file where prompted. **Note: Only .wav files are supported**
- Run!

Running `deepNOID.py` should take ~15s for a 3min track.

As this is my first go at this, the model is probably not amazingly optimised and warnings may appear. Please feel free to PR optimisations either in the code or model implementation! Running `deepNOID.py` should take ~15s for a 3min track.


## How It Works / Project Layout

### Dataset Preparation

Every thing to do with the dataset preparation in this project is located in the `training_data_and_prep` folder. Raw audio data had to be split into tracks with equal length,  MFCC's had to be extracted for each segment of each track and finally, MFCC data and labels all had to be written to a JSON file which our Tensorflow model could utilise. 

Firstly, 20 full length Death Grips tracks were placed into `raw_training_data/noid` and 20 full length tracks of unnoided material placed in `raw_training_data/not_noid`. The unnoided material includes, but isn't limited to, Justin Bieber, 14th Century Choral music, Falling In Reverse, Pachelbel's Canon in D and Joe Rogan talking about how smart people are dumb for 3 minutes. Not going to lie, this was one of the most fun parts of the project.

`training_data_and_prep/wav_join.py` is run to combine all .wav files in a directory into one long continuous track. `training_data_and_prep/wav_split.py` is this used to split this one long .wav file into tracks of equal length. For this model, the tracks were split into 30s snippets.

`training_data_and_prep/data_generation` is then run on the `raw_training_data` folder (after .wav files have been adjusted for length). This takes the dataset path and a path named `JSON_PATH` which is where the MFCC data and labels will be written to. The process of MFCC extraction is shown further in the comments.

### Model Creation

The RNN-LSTM model generation was done using `lstm_model_generation.py`. This loads the data from the JSON file created previously, splits it into training, test and validation data and builds a sequencial model using 2 LSTM layers, 2 dense layers and an output layer. 

The LSTM layers and first dense layer have 64 neurons, the second dense layer 16 and the output layer has 2. The dropouts for the two dense layers were 0.4 and 0.3 respectively and both utilised 'relu' activation functions. The output uses a 'softmax' activation to provide the probability of a track being noided ot not noided.

This routinely achieved ~95% validation accuracy after 100 epochs using a batch size of 32 and a learning rate of 0.0001. This model is saved in the `models` directory.


### Model Predicitons

Running `deepNOID.py` on a .wav file (of any length) will make a call to the prediction function in `classification_prediction.py`. This prediction function will split the track into 25 sections and call the preprocessing function within `audio_preprocessing.py` on each. This function extracts MFCC data in the same way it was extracted in training and returns this data to the prediction funciton. Each sections MFCC data is put through Tensorflow's `model.predict` and the result is appended to `noid_vec`, a list of the results for each section (0=noided, 1=not noided). The label which appears most in this list once the process is ended is deduced to be the result and a message is printed to the user.

**Note: The sections of the user supplied track are temporarily stored in `temp_storage` as MFCC data is being extracted. These track sections are deleted before the prediction process ends**.


## Acknowledgments

* This project is heavily based off [Valerio Velardo's "Deep Learning (for audio) with Python"](https://www.youtube.com/watch?v=fMqL5vckiU0&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf) course on YouTube. He is easily one of the best online teachers I've ever come across, explaining things in a simple, concise yet massively informative way and if you want to get into deep learning with audio his channel is an absolute goldmine.
