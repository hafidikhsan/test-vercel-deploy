# Install Library
import cloudinary
import cloudinary.uploader
import cloudinary.api
import logging
import os
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
from flask import jsonify
from flask import Flask, render_template, request
from cloudinary.utils import cloudinary_url

# Library Fluency
import numpy as np
import tensorflow as tf
import librosa

# Cloudinary API
CLOUD_NAME='dntqqcuci'
API_KEY='182168662843965'
API_SECRET='3M7PhAHGCMOOr5EcImPt1g-bxvw'

# Konfigurasi Flask
load_dotenv()

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Konfigurasi Cloudinary
cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=API_KEY,
    api_secret=API_SECRET
)

# Verifikasi Cloud
app.logger.info('%s', 'dntqqcuci')

#Fluency - Model path
fluency_model_dir = '..\assets\model\fluency_model_real_data'

# Fluency - Fungsi ekstraksi ciri
def feature_extraction(file_name):
    X , sample_rate = librosa.load(file_name)

    if X.ndim > 1:
        X = X[:,0]
    X = X.T
            
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=30).T, axis=0)
    rmse = np.mean(librosa.feature.rms(y=X).T, axis=0)
    spectral_flux = np.mean(librosa.onset.onset_strength(y=X, sr=sample_rate).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)

    return mfccs, rmse, spectral_flux, zcr

#Fluency - Load Model
fluency_modelss = tf.keras.models.load_model(fluency_model_dir)

# Upload File Fluency
@app.route("/fluency", methods=['POST'])
@cross_origin()
def upload_file_fluency():
  app.logger.info('in upload route')
  print('API', API_KEY)

  cloudinary.config()
  upload_result = None

  if request.method == 'POST':
    file_to_upload = request.files['file']
    app.logger.info('%s file_to_upload', file_to_upload)

    if file_to_upload:
      upload_result = cloudinary.uploader.upload(file_to_upload, resource_type="video")
      app.logger.info(upload_result)
      app.logger.info(type(upload_result))
      print(upload_result['secure_url'])
      
      # Get data
      mfccs, rmse, spectral_flux, zcr = feature_extraction(upload_result['secure_url'])

      # Make array
      number_of_features = 3 + 30
      datasetcheck = np.empty((0,number_of_features))
      
      # Data to array
      extracted_features = np.hstack([mfccs, rmse, spectral_flux, zcr])
      datasetcheck = np.vstack([datasetcheck, extracted_features])

      # Expand dimension
      datasetcheck = np.expand_dims(datasetcheck, axis=1)
      
      # Lexical - Melakukan prediksi
      pred = fluency_modelss.predict(datasetcheck)
      classes = np.argmax(pred, axis = 1).tolist()
      print(classes)
      return jsonify({"Fluency Class": classes[0]})


# Optimisasi Cloud
@app.route("/cld_optimize", methods=['POST'])
@cross_origin()
def cld_optimize():
  app.logger.info('in optimize route')

  cloudinary.config()
  if request.method == 'POST':
    public_id = request.form['public_id']
    app.logger.info('%s public id', public_id)
    
    if public_id:
      cld_url = cloudinary_url(public_id, fetch_format='auto', quality='auto', secure=True)
      app.logger.info(cld_url)
      return jsonify(cld_url)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'