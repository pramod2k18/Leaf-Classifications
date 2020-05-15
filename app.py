
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

#Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init():
    global graph
    graph = tf.compat.v1.get_default_graph()

def read_image(filename):
    
    img = load_img(filename, grayscale=False, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32')
    img = img / 255.0
    return img


@app.route('/', methods=['GET', 'POST'])
def main_page():

        return render_template('index.html')


@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        try:
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join('uploads', filename)
                file.save(file_path)
                img = read_image(file_path)
                

                with graph.as_default():
                    model1 = load_model("Leaf_vs_Non-Leaf.h5")
                    predict1 = model1.predict(img)
                    class_prediction1 = np.argmax(predict1,axis=1)

                    if class_prediction1[0] == 0:
                        model2 = load_model("leaf_vgg.h5")
                        predict2 = model2.predict(img)
                        class_prediction2=np.argmax(predict2,axis=1)
                        
                        if class_prediction2[0] == 0:
                            product1 = "Apple"
                            product2 = "আপেল"
                        elif class_prediction2[0] == 1:
                            product1 = "Bluebery"
                            product2 = "ব্লুবেরি"
                        elif class_prediction2[0] == 2:
                            product1 = "Pepper"
                            product2 = "কেপছিকাম"
                        elif class_prediction2[0] == 3:
                            product1 = "Soybean"
                            product2 = "সয়াবিন"
                        else:
                            product1 = "Tomato"
                            product2 = "বিলাহী"
                        
                    else:
                        product1 = "This is not a Leaf"
                        product2 = "Non-Leaf"
                return render_template('predict.html', product1 = product1, product2 = product2)
                                        
        except Exception as e:
            return "Unable to read the file. Please check if the file extension is correct."

    return render_template('predict.html')

if __name__ == "__main__":
    init()
    app.run()




