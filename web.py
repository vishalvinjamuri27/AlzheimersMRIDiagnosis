import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load your trained model
model = tf.keras.models.load_model('best_weights.hdf5')

# Create a function to predict the class of an MRI scan
def predict(image_path):
  # Read and preprocess the image
  img = image.load_img(image_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = x / 255.0

  # Use the model to predict the class
  predictions = model.predict(x)
  class_idx = np.argmax(predictions[0])

  # Map the class index back to the original class names
  classes = ['mildly demented', 'moderately demented', 'non-demented', 'very mildly demented']
  return classes[class_idx]

# Create the main page
def main():
  st.title('MRI Scan Diagnosis')
  st.sidebar.header('Input')
 
  # Allow the user to upload an MRI scan
  uploaded_file = st.sidebar.file_uploader('Choose an MRI scan', type=['jpg', 'png'])
  if uploaded_file is not None:
    # Predict the class of the MRI scan
    prediction = predict(uploaded_file)
    st.sidebar.success(f'Prediction: {prediction}')

if __name__ == '__main__':
  main()
