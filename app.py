from flask import Flask, render_template, request, session, redirect, url_for, flash
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
#from keras.utils.generic_utils import get_custom_objects
from keras_preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
import matplotlib.pyplot as plt
import cv2
#import numpy as np
import csv
#import diseaseprediction
import pandas as pd
import random

UPLOAD_FOLDER = './flask app/assets/images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask app/assets', 
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = 'static'

model = pickle.load(open('Liver2.pkl', 'rb'))



@app.route('/')
def root():
   return render_template('index.html')

@app.route('/validate',methods=['GET', 'POST'])
def validate():
  if request.method=='POST':
     u=str(request.form['username'])
     p=str(request.form['password'])
     
     if u=="Admin" and p=="admin123":
        return render_template('index.html',u=u)
     else:
        msg="Error message:Please Enter a valid user name or password"
        return render_template('login.html',msg=msg)
     


@app.route('/index.html',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route('/numeric_data_pred.html')
def NumericDataPred():
    return render_template('numeric_data_pred.html')


@app.route('/upload.html')
def upload():
   return render_template('upload.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Age = int(request.form['Age'])
        Gender = int(request.form['Gender'])
        Total_Bilirubin = float(request.form['Total_Bilirubin'])
        Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
        Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
        Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
        Total_Protiens = float(request.form['Total_Protiens'])
        Albumin = float(request.form['Albumin'])
        Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])


        values = np.array([[Age,Gender,Total_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])
        prediction = model.predict(values)

        return render_template('result.html', prediction=prediction)


@app.route('/uploaded_ct', methods = ['POST', 'GET'])
def uploaded_ct():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_ct.jpg'))

   resnet_ct = load_model('models/resnet_ct.h5')
   vgg_ct = load_model('models/vgg_ct.h5')

   image = cv2.imread('./flask app/assets/images/upload_ct.jpg') # read file 
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
   image = cv2.resize(image,(224,224))
   image = np.array(image) / 255
   image = np.expand_dims(image, axis=0)
   
   resnet_pred = resnet_ct.predict(image)
   probability = resnet_pred[0]
   print("Resnet Predictions:")
   if probability[0] > 0.5:
      resnet_ct_pred = str('%.2f' % (probability[0]*100) + '% Liver Disease') 
   else:
      resnet_ct_pred = str('%.2f' % ((1-probability[0])*100) + '% Non Liver Disease')
   print(resnet_ct_pred)

   vgg_pred = vgg_ct.predict(image)
   probability = vgg_pred[0]
   print("VGG Predictions:")
   if probability[0] > 0.5:
      vgg_ct_pred = str('%.2f' % (probability[0]*100) + '% Liver Disease') 
   else:
      vgg_ct_pred = str('%.2f' % ((1-probability[0])*100) + '% Non Liver Disease')
   print(vgg_ct_pred)

   return render_template('results_ct.html',resnet_ct_pred=resnet_ct_pred,vgg_ct_pred=vgg_ct_pred)


if __name__ == "__main__":
    app.run(debug=True)

