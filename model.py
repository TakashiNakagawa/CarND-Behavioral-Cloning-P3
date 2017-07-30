import csv
import cv2
import numpy as np
import sklearn.utils
from sklearn.model_selection import train_test_split


def get_samples():
    '''get samples from csv file.
    Returns:
        train and validation samples
    '''
    lines = []
    with open('../data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    return train_test_split(lines, test_size=0.2)


train_samples, validation_samples = get_samples()


def flip_image_angle(image, angle):
    '''
    flip image and angle
    :param image: image
    :param angle: angle
    :return: flipped image and angle
    '''
    return cv2.flip(image, 1), angle * -1.0


def read_image(path):
    '''
    read image from csv
    :param path: csv path
    :return: image
    '''
    name = '../data/IMG/' + path.split('/')[-1]
    return cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)


def read_image_angle(batch_sample, camera_type):
    '''
    read image and angle from csv file.
    :param batch_sample: csv column
    :param camera_type: center, left or right
    :return: image and angle
    '''
    correction = 0.2
    angle = float(batch_sample[3])
    if camera_type == "center":
        return read_image(batch_sample[0]), angle
    elif camera_type == 'left':
        return read_image(batch_sample[1]), angle + correction
    else:
        return read_image(batch_sample[2]), angle - correction

def generator(samples, use_augmented_image, batch_size=32):
    '''
    generator for keras
    :param samples: target samples
    :param use_augmented_image: augment image or not
    :param batch_size: batch size
    :return:
    '''
    num_samples = len(samples)
    samples = sklearn.utils.shuffle(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                camera = np.random.choice(['center', 'left', 'right'])
                if not use_augmented_image:
                    image, angle = read_image_angle(batch_sample, 'center')
                else:
                    image, angle = read_image_angle(batch_sample, camera)
                    if np.random.random() > 0.5:
                        image, angle = flip_image_angle(image, angle)
                images.append(image)
                angles.append(angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, True, batch_size=32)
validation_generator = generator(validation_samples, False, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
import keras.optimizers

def create_model():
    '''
    create model
    :return: model
    '''
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Convolution2D(24, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(36, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(48, 5, 5))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    return model

model = create_model()
opt = keras.optimizers.Adam(lr=0.001)  # lr=0.001
model.compile(loss='mse', optimizer=opt)
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                     validation_data=validation_generator, nb_val_samples=len(validation_samples),
                                     nb_epoch=5, verbose=1)

model.save('model.h5')

print(history_object.history.keys())

import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('graph.png')
plt.show()
