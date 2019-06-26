from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model


import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/'
model_1 = load_model('data/im_model.h5')
model_1._make_predict_function() 


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():


    if request.method == 'POST':
        image_file = request.files['image_file']
        filename = secure_filename(image_file.filename)
        image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        
        img = image.load_img(image_file, target_size=(224, 224))
        

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x, mode='caffe')
        preds = model_1.predict(x)
        df = pd.DataFrame(decode_predictions(preds, top=3)[0])
        fig, ax = plt.subplots(1, 1, dpi=100)
        sns.barplot(df[2]*100, df[1])
        ax.set_ylabel('')    
        ax.set_xlabel('Probability')
        fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'],'plot.png'))

    return render_template('result.html', url_image = os.path.join(app.config['UPLOAD_FOLDER'], filename), url_graph = os.path.join(app.config['UPLOAD_FOLDER'], 'plot.png'))


if __name__ == '__main__':
    app.run()
