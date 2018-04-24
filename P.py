# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

try:
    open ("model.json", "r")
except IOError:
    
    # Initialising the CNN
    print("Initialising the CNN")
    classifier = Sequential()
    
    # Step 1 - Convolution
    print("Step 1 - Convolution")
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    
    # Step 2 - Pooling
    print("Step 2 - Pooling")
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a second convolutional layer
    print("Adding a second convolutional layer")
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Step 3 - Flattening
    print("Step 3 - Flattening")
    classifier.add(Flatten())
    
    # Step 4 - Full connection
    print("Step 4 - Full connection")
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
    # Compiling the CNN
    print("Compiling the CNN")
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
    
    # Part 2 - Fitting the CNN to the images
    print("Part 2 - Fitting the CNN to the images")
    from keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory('dataset/training_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')
    
    test_set = test_datagen.flow_from_directory('dataset/test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')

    # adjust these to change the number of epochs and 
    # steps for training and testing
    s_per_epoch = 5000
    n_epoch = 15
    v_steps = 2000
    
    classifier_metrics = classifier.fit_generator(training_set,
    steps_per_epoch = s_per_epoch,
    epochs = n_epoch,
    validation_data = test_set,
    validation_steps = v_steps)
    
    # get the various metrics to be recorded
    acc = classifier_metrics.history['acc']
    loss = classifier_metrics.history['loss']
    val_acc = classifier_metrics.history['val_acc']
    val_loss = classifier_metrics.history['val_loss']
    
    # init counter
    i = 0
    
    metrics_file = open("metrics.txt", 'w')
    
    #  write metrics to file
    while i < n_epoch:
        i += 1
        metrics_file.write(str("Epoch " + str(i) +":"))
        metrics_file.write(str("\taccuracy: " + str(acc[i-1])))
        metrics_file.write(str("\tloss: " + str(loss[i-1])))
        metrics_file.write(str("\tvalidaton accuracy: " + str(val_acc[i-1])))
        metrics_file.write(str("\tvalidation loss: " + str(val_loss[i-1])))
        metrics_file.write("\n")
    
    metrics_file.close()
        

    
    #plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    # Part 2.5 - saving the model to disk
    model = classifier.to_json()
    with open("model.json", "w") as model_json_file:
        model_json_file.write(model)
    classifier.save_weights("model.h5")
    print("Saved model to disk")

# Part 3 - Loading model from disk
model_json_file = open("model.json", "r")
loaded_model_json = model_json_file.read()
model_json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Part 3.5 - Making new predictions
print("Part 3 - Making new predictions")
import numpy as np
from keras.preprocessing import image
# ask for number of predicitons to make
test_image_number = input('Enter the number of predictions you would like to make: ')
# set counter
i = 0
# grab a random image from the prediction dataset until all predictions are made
while i < int(test_image_number):
    i += 1

# old image loading code, remove when above loop is implemented
test_image = image.load_img('dataset/single_prediction/neither1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
# this needs to be changed I think, see https://keras.io/models/sequential/ and the predict function
test_image = np.expand_dims(test_image, axis=0)
result = loaded_model.predict(test_image)

print(result[0][0])

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print("\n\nPREDICTION: ", prediction)