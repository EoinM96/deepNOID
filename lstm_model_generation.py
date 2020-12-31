import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import json

# Where to read training data from
DATA_PATH = 'training_data_and_prep/usable_data.json'


def load_data(data_path):
    """
    Loads JSON data into numpy arrays

    :param data_path: Path supplied by user at top of script
    :return: MFCC's and associated labels as numpy arrays
    """
    with open(data_path, 'r') as fp:
        data = json.load(fp)

    X = np.array(data['mfcc'])
    y = np.array(data['labels'])
    return X, y


def prep_datasets(test_size, validation_size):
    """
    Splits training data into training, test and validation sets.

    :param test_size: Relative size of training data to use for testing
    :param validation_size: Relative size of remaining training data to use as validation testing
    :return:
    """
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_test, X_validation, y_train, y_test, y_validation


def build_model(input_shape):
    """
    Creates the RNN-LSTM model

    :param input_shape: Dimension of input shape from X data
    :return: Tensorflow2 RNN-LSTM model
    """
    model = tf.keras.Sequential()

    # 2 LSTM layers
    model.add(tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.LSTM(64))

    # 2 dense + output layer
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    return model


if __name__ == '__main__':
    # Split training data
    X_train, X_validation, X_test, y_train, y_validation, y_test = prep_datasets(0.25, 0.2)

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (130(?), 13)
    model = build_model(input_shape)

    # Hyper parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Save model history
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=100)

    # Print error and accuracy
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print('\nAccuracy on Test Set: {}'.format(test_accuracy))

    # Save model
    model.save('F:/TF_tutorial/models/')
