import os
from flask import Flask, request, redirect, url_for, render_template
#from werkzeug.utils import secure_filename

app = Flask(__name__)

import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.applications import VGG16
import numpy as np
from keras.preprocessing import image

#to remove CUDA Error
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



conv_base = VGG16(weights='imagenet', 
                      include_top=False,
                      input_shape= (224, 224, 3))

global sess
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)


def init():
    global graph
    graph = tf.compat.v1.get_default_graph()

global model

b = load_model('Model1.h5')

@app.route('/prediction/<filename>')
def prediction(filename):
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_image(file.filename):
                filename = file.filename
    
                a = os.path.join('uploads', filename)
                
                img = image.load_img(a, target_size=(224, 224))
                img_tensor = image.img_to_array(img)  
                img_tensor /= 255.  
                
                p = img_tensor.reshape(1,224, 224, 3)
                features = conv_base.predict(p)
                
                try:
                        prediction = b.predict(features)
                except:
                        prediction = b.predict(features.reshape(1, 7*7*512))
                
                with graph.as_default():
                    tf.compat.v1.keras.backend.set_session(sess)        
                    classes = ["Apple", "Blueberry", "Pepper Bell", "Soybean", "Tomato"]
                    classesAs = ["আপেল",  "ব্লুবেরি",  "কেপছিকাম",  "সয়াবিন", "বিলাহী"]
                    
                    x = np.array(prediction[0])
                    y = np.argsort(x)
                    
                    predictions = {
                            "class1":classes[y[4]],
                            "class2":classesAs[y[4]]
                    }
                
        else:
                print("That file extension is not allowed")
                return redirect(url_for('prediction', filename=filename))
            
    return render_template('predict.html', predictions= predictions)

if __name__ == '__main__':
    init()
    app.debug = True
    app.run() 