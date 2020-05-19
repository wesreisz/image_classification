from flask import Flask
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import xception
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils

from PIL import Image
import flask
import io
import requests

MODEL_LOCATION = '/opt/section/xception_weights_tf_dim_ordering_tf_kernels.h5'

application = Flask(__name__)
model = None

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
    global model
    model=tf.keras.applications.xception.Xception(weights=MODEL_LOCATION,include_top=True)

def prepare_image(image):
    #make sure image is rgb
    if image.mode != "RGB":
        image = image.convert("RGB")
   
    #set image size to 299 x 299
    image = image.resize((299, 299))

    image=tf.keras.preprocessing.image.img_to_array(image)
    image=tf.keras.applications.xception.preprocess_input(image)
    return image

@application.route("/", methods=["GET"])
def index():
    return """
    <h3>API Server</h3>
    <p>
        Available methods:
        <ul>
            <li>/ [GET]: This message</li>
            <li>/status [GET]: gets OK status</li>
            <li>/predict [POST]: Classifies an image</li>
        </ul>
    </p>
    """

@application.route("/status", methods=["GET"])
def status():
    return '{"status":"ok"}'

@application.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image)
            
            # make prediction
            predictions=model.predict(np.array([image]))
            results = xception.decode_predictions(predictions,top=5)
            
            # build result
            data["predictions"] = []
            for (_, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

#if model has not previously been set, load it.
if model == None:
    print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
    load_model()

if __name__ == "__main__":
    application.run(host='0.0.0.0', port='5000')
    
    