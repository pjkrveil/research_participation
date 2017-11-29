from keras import applications, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input
import keras

print("++ keras version is: v.")
print(keras.__version__)
# OUTPUT: '2.0.8'

keras.backend.set_image_dim_ordering('tf')


# SET ALL THE PARAMETERS
weights_path = 'models/vgg16.h5'
img_width, img_height = 224, 224
train_data_dir = '/data/pjkrveil/train_val_images'
validation_data_dir = '/data/pjkrveil/train_val_images'
nb_train_samples = 500000
nb_validation_samples = 79184
epochs = 50
batch_size = 50000

# LOAD VGG16
input_tensor = Input(shape=(224, 224, 3))
model = applications.VGG16(weights='imagenet',
                           include_top=False,
                           input_tensor=input_tensor,
                           classes=5089)


# Truncate and replace softmax layer for transfer learning
model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []
model.layers.add(Dense(5089, activation='softmax', name='predictions'))

# Learning rate is changed to 0.001
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


# CREATE AN "REAL" MODEL FROM VGG16
# BY COPYING ALL THE LAYERS OF VGG16
new_model = Sequential()
for l in model.layers:
    new_model.add(l)

# LOCK THE TOP CONV LAYERS
for layer in new_model.layers:
    layer.trainable = False

# COMPILE THE MODEL
sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
new_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])


# CREATE THE IMAGE GENERATORS
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                        train_data_dir,
                        target_size=(img_height,img_width),
                        batch_size=batch_size,
                        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
                            validation_data_dir,
                            target_size=(img_height,img_width),
                            batch_size=batch_size,
                            class_mode='binary')


#  FIT THE MODEL

new_model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)
