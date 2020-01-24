import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
#

path = "./data/chest_xray"
dirs = os.listdir(path)

train_folder = f"{path}/train/"
test_folder = f"{path}/test/"
val_folder = f"{path}/val/"

train_normal = f"{train_folder}NORMAL/"
train_pneu = f"{train_folder}PNEUMONIA/"

normal_images = glob(f"{train_normal}*.jpeg")
pneu_images = glob(f"{train_pneu}*.jpeg")

def show_imgs(num_of_imgs: int):
    for img in range(num_of_imgs):
        pneu_pic = np.asarray(plt.imread(pneu_images[img]))
        normal_pic = np.asarray(plt.imread(normal_images[img]))

        fig = plt.figure(figsize = (15,10))
        normal_plot = fig.add_subplot(1,2,1)
        plt.imshow(normal_pic, cmap='gray')
        normal_plot.set_title('Normal')
        plt.axis('off')

        pneu_plot = fig.add_subplot(1,2,2)
        plt.imshow(pneu_pic, cmap='gray')
        pneu_plot.set_title('Pneumonia')
        plt.axis('off')

        plt.show()

train_datagen = ImageDataGenerator(rescale = 1/255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)

test_datagen = ImageDataGenerator(rescale = 1/255)

training_set = train_datagen.flow_from_directory(train_folder,
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

val_set = test_datagen.flow_from_directory(val_folder,
                                           target_size=(64,64),
                                           batch_size=32,
                                           class_mode='binary')

test_set = test_datagen.flow_from_directory(test_folder,
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_train = model.fit_generator(training_set,
                         steps_per_epoch = 200,
                         epochs = 25,
                         validation_data = val_set,
                         validation_steps = 100)

test_accuracy = model.evaluate_generator(test_set,steps=624)

print('Testing Accuracy: {:.2f}%'.format(test_accuracy[1] * 100))


