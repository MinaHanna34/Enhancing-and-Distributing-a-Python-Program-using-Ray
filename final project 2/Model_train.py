from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize the CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Image Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

# Training set
training_set = train_datagen.flow_from_directory('C:/Users/minat/Downloads/chest_xray/chest_xray/train', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

# Test set
test_set = test_datagen.flow_from_directory( "C:/Users/minat/Downloads/chest_xray/chest_xray/test", target_size = (64, 64), batch_size = 32, class_mode = 'binary')

# Training the model
classifier.fit(training_set, steps_per_epoch = int(5216/32), epochs = 25, validation_data = test_set, validation_steps = int(624/32))

# Save model
classifier.save('pneumonia_model.h5')
print("Model saved as pneumonia_model.h5")
