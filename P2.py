# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import json
from PIL import Image
from os import walk
from keras.utils.vis_utils import plot_model
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

labels = {}

# Open the model to be read if it exists
try:
    open("model2.json", "r")

# Create it if it doesn't exists
except IOError:

    # Initialising the CNN
    print("Initialising the CNN")
    classifier = Sequential()

    # Step 1 - Convolution
    print("Step 1 - Convolution")
    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

    # Step 2 - Pooling
    print("Step 2 - Pooling")
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a second convolutional layer
    print("Adding a second convolutional layer")
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 3 - Flattening
    print("Step 3 - Flattening")
    classifier.add(Flatten())

    # Step 4 - Full connection
    print("Step 4 - Full connection")
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=3, activation='softmax'))

    # Compiling the CNN
    print("Compiling the CNN")
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    # Part 2 - Fitting the CNN to the images
    print("Part 2 - Fitting the CNN to the images")
    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='categorical')
    print(training_set.class_indices)
    print("###############")
    labels = dict((y, x) for x, y in training_set.class_indices.items())
    print(labels)
    labels_file = open("labels.txt", 'w')
    labels_file.write(json.dumps(labels))

    print("###############")
    print(type(training_set.classes))
    headersList = [labels.get(item, item) for item in training_set.classes]
    print(headersList)

    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='categorical')

    # adjust these to change the number of epochs and
    # steps for training and testing
    s_per_epoch = 6
    n_epoch = 1
    v_steps = 2000

    classifier_metrics = classifier.fit_generator(training_set,
                                                  steps_per_epoch=s_per_epoch,
                                                  epochs=n_epoch,
                                                  validation_data=test_set,
                                                  validation_steps=v_steps)


    # get the various metrics to be recorded
    acc = classifier_metrics.history['acc']
    loss = classifier_metrics.history['loss']
    val_acc = classifier_metrics.history['val_acc']
    val_loss = classifier_metrics.history['val_loss']

    # init counter
    i = 0

    # Open 2nd metrics file to write to
    metrics_file = open("metrics2.txt", 'w')

    #  write metrics to file
    while i < n_epoch:
        i += 1
        metrics_file.write(str("Epoch " + str(i) + ":"))
        metrics_file.write(str("\taccuracy: " + str(acc[i - 1])))
        metrics_file.write(str("\tloss: " + str(loss[i - 1])))
        metrics_file.write(str("\tvalidaton accuracy: " + str(val_acc[i - 1])))
        metrics_file.write(str("\tvalidation loss: " + str(val_loss[i - 1])))
        metrics_file.write("\n")

    metrics_file.close()

    # plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # Part 2.5 - saving the model to disk
    model = classifier.to_json()
    with open("model2.json", "w") as model_json_file:
        model_json_file.write(model)
    classifier.save_weights("model2.h5")
    print("Saved model to disk")

# Part 3 - Loading model from disk
model_json_file = open("model2.json", "r")

# Read model
loaded_model_json = model_json_file.read()

# Close model
model_json_file.close()

# Store loaded model
loaded_model = model_from_json(loaded_model_json)

# Load weights
loaded_model.load_weights("model2.h5")
print("Loaded model from disk")

# Part 3.5 - Making new predictions
print("Part 3 - Making new predictions")
import numpy as np
from keras.preprocessing import image

# String for user input
cont = '0'
index = 0

print("\n\n")

# Open labels file
with open("labels.txt") as f:
        for line in f:
            labels = json.loads(line)

# Display user picture options
while cont == '0':
    for (dirpath, dirnames, filenames) in walk('dataset/single_prediction/'):
        print("\n\nPREDICTION FILES")
        for fn in filenames:
            print(index,fn)
            index += 1
        break

    # Reads user input to allow choice of picture
    while True:
        test_image_number = input('\nEnter a valid number of the image you would like to predict on: ')
        if int(test_image_number) > 0 and int(test_image_number) < 13:
            break
    
    # Store filename based on user input
    fileName = 'dataset/single_prediction/'+filenames[int(test_image_number)]
    
    # Load image user choice
    test_image = image.load_img(fileName, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    
    # this needs to be changed I think, see https://keras.io/models/sequential/ and the predict function
    test_image = np.expand_dims(test_image, axis=0)
    
    # Store the prediction
    result = loaded_model.predict(test_image)
    result2 = loaded_model.predict_classes(test_image)
    
    # Print out the result
    print(result)
   
    # Print out the prediction choice
    print(result2)

    # Print the array of options
    print(labels)

    # Print the final prediction
    print("\n\nPREDICTION: ", labels.get(str(result2[0])))
    
    # Open the image that was used
    img = Image.open(fileName)
    
    # Show it on screen to the user
    img.show()

    # Continue to prompt user until they enter something that isn't 0
    cont = input("\nMake more predictions? (type 0 for yes or anything else for no): ")

    # Strip user input
    cont = cont.strip()
    
    # Reset index to 0
    index = 0
