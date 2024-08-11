# install streamlit for Using it the model as App
from keras_vggface.utils import preprocess_input
from keras_vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))


def save_uploaded_image(uploaded_img):
    try:
        with open(os.path.join("uploads", uploaded_img.name), 'wb') as f:
            f.write(uploaded_img.getbuffer())
        return True
    except:
        return False


def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    # Extract features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()
    return result


def recommend(features, feature_list):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos


st.title('Face Detector Recommender')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # saving image into directory
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)
        # extracting features from image
        features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)
        # recommended image
        index_pos = recommend(features, feature_list)
        # st.text(index_pos)
        predicted_face = " ".join(filenames[index_pos].split('\\')[1].split('_'))
        #display
        col1, col2 = st.columns(2)

        with col1:
            st.header('Your Uploaded image')
            st.image(display_image)
        with col2:
            st.header('Matches with ' + predicted_face)
            st.image(filenames[index_pos],width=400)
        
