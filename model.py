from keras import applications, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input
import keras
import theano

# SETTING FOR USING GPU
theano.config.device = 'gpu'
theano.config.floatX = 'float32'

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


# CREATE A TOP MODEL
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
top_model.load_weights(top_model_weights_path)


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
new_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


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
