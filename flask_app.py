from CGAN.generator import Generator

from flask import Flask, request, send_file, abort
from werkzeug.exceptions import HTTPException

import torch
from torchvision import transforms

from io import BytesIO

valid_classes = ["T-shirt/top","Trouser", "Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
# Load and set generator parameters
latent_dim = 100
n_classes = 10
image_shape = [28, 28]

model = Generator(latent_dim, n_classes, image_shape)
model.load_state_dict(torch.load('saved_models/generator_weights.pth'))
model.eval()

#Creating Flask app 
app = Flask(__name__)
app.config["DEBUG"] = False

def serve_pil_image64(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', subsampling=0, quality=95)
    return img_io


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        #Get the input label
        label = request.form.get('label')
        if label in valid_classes:
            #Get the image a tensor
            prediction_img = get_prediction(transform_input(label))

            #Transform the image into a binary object and send it back as a response in jpeg format
            pil_img = transforms.ToPILImage()(prediction_img)
            response = serve_pil_image64(pil_img)
            return send_file(BytesIO(response.getvalue()),mimetype='image/jpeg')
        else:
            #Abort the program with 400 in case of mispelling in the curl command
            abort(400)

#General error handler
@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e

    res = {'code': 500,
           'errorType': 'Internal Server Error',
           'errorMessage': "Something went really wrong!"}
    
    return res

#Get the user input and transform it to its repsective class number
def transform_input(label):
    id2class = {
            "T-shirt/top": 0,
            "Trouser":1,
            "Pullover":2,
            "Dress":3,
            "Coat":4,
            "Sandal":5,
            "Shirt":6,
            "Sneaker":7,
            "Bag":8,
            "Ankle boot":9,}   
    return id2class[label]

#Use the generator model to generate the image
def get_prediction(label):
    prediction = model(torch.randn(1, latent_dim), torch.tensor([label]))
    return prediction
