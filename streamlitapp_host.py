import streamlit as st # type: ignore
import tensorflow as tf
import numpy as np
import cv2  # type: ignore
from tensorflow.keras.applications.densenet import preprocess_input as DenseNet_preprocess_input 

st.header("Image Class Predictor")
### Load the classifier model
model = tf.keras.models.load_model('saved_model/desnet_bestmodel.hdf5', compile=False)
### Load file to be classified
uploaded_file = st.file_uploader("Choose an image file", type=["PNG", "JPG", "JPEG"])

### Map the class name with the class indices
map_dict= {0: 'benign', 1: 'malignant', 2: 'normal'}

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224, 224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = DenseNet_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Predict Image")    
    if Genrate_pred:
        
        prediction = model.predict(img_reshape).argmax()
        #st.title(prediction.astype(str))
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))