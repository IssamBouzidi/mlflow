# import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.preprocessing import image
import numpy as np


# def resize_image(img, size=(20,20)):

#     h, w = img.shape[:2]
    
#     if h == w: 
#         return cv2.resize(img, size, cv2.INTER_AREA)

#     dif = h if h > w else w


#     if dif > (size[0] + size[1]):
#         interpolation = cv2.INTER_AREA
#     else:
#         interpolation = cv2.INTER_CUBIC

#     x_pos = (dif - w)//2
#     y_pos = (dif - h)//2

#     mask = np.zeros((dif, dif), dtype=img.dtype)
#     mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]

#     return cv2.resize(mask, size, interpolation)


def get_categories(dir_path):
    categories = []
    for root, subdirectories, files in os.walk(dir_path):
        for subdirectory in subdirectories:
            categories.append(subdirectory)
    
    return categories


def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def earlystop(nbr_patience):
    return EarlyStopping(patience=nbr_patience)


def learning_rate_reduction():
    return ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)


def train_model(model, epochs, train_generator, validation_generator, callbacks = None):
    return model.fit_generator(train_generator, 
                                epochs=epochs, 
                                validation_data=validation_generator,
                                callbacks = callbacks)


def loss_visualisation(training_loss, training_val_loss):
    plt.plot(training_loss, color='red', label='Training loss')
    plt.plot(training_val_loss,  color='green', label='Validation loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.legend()

    plt.show()


def accuracy_visualisation(training_accuracy, training_val_accuracy):
    plt.plot(training_accuracy, color='red', label='Training accuracy')
    plt.plot(training_val_accuracy, color='green', label='Validation accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.legend()

    plt.show()


def init_activation(model, test_img):
    layer_outputs = [layer.output for layer in model.layers]
    img = image.load_img(test_img, target_size=(32,32, 3))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    return activation_model.predict(img_arr)


def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*13.5,col_size*2.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='plasma')
            activation_index += 1



from keras.preprocessing import image
from keras.models import load_model
import pandas as pd
import seaborn as sns


def predict_image(model, categories, choix):
    test_image = image.load_img(choix, target_size = (32, 32))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)

    preds = model.predict_classes(test_image)
    prob = model.predict_proba(test_image)

    index = preds[0]
    print(f'Categorie {categories[preds[0]]}, Prédiction {"%.2f" % (prob[0][index] * 100)}%')
    
    predictrions_array = []
    
    for x in range(0,2):
        predictrions_array.append([categories[x], prob[0][x]])
    
    df = pd.DataFrame(predictrions_array, columns = ['Category', 'Prediction'])

    f, axarr = plt.subplots(1,2, figsize=(10,4))

    img = cv.imread(choix)
    axarr[0].imshow(img)
    axarr[0].axis('off')

    axarr[1] = sns.barplot(x="Prediction", y="Category", data=df)
    sns.set_style(style='white')

    axarr[1].set_ylabel('Category')    
    axarr[1].set_xlabel('Prediction')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    f.suptitle("Model Prediction")
    f.subplots_adjust(top=0.88)