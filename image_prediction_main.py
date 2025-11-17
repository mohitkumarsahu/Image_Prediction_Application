from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

import numpy as np

# load the vgg16 pre-trained model
model=VGG16(weights='imagenet', include_top=True,input_shape=(224,224,3))


#Loading the image and preprocess it for the model

img_path=r"C:\Users\HP\Desktop\l.jpg"
img=image.load_img(img_path, target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)         #Add batch dimension to image so shape become (1,224,224,3)
x=preprocess_input(x)              #because keras expect image in batch format.

#Using the model to predict the class
pred=model.predict(x)

decode_prediction=decode_predictions(pred, top=1)[0]

for pred in decode_prediction:
    print(pred[1], ":   ", pred[2] )