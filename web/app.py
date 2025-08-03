from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from pymongo import MongoClient
import bcrypt
import numpy as np
import requests

from keras.applications import InceptionV3, imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from io import BytesIO


app = Flask(__name__)
api = Api(app)

# Load pre-trained InceptionV3 model
pretrained_model = InceptionV3(weights="imagenet")

client = MongoClient("mongodb://db:27017")

# Create a new collection for image recognition
db = client.ImageRecognition
users = db["Users"]


def user_exists(username):
    """Check if a user already exists in the database."""
    return users.find_one({"username": username}) is not None


class Register(Resource):
    def post(self):
        # We first get the JSON data from the request
        data = request.get_json()

        # Get username and password from the data
        username = data.get("username")
        password = data.get("password")

        if user_exists(username):
            return jsonify({"status": 301, "message": "User already exists"})

        # If user is new, we hash the password and store it
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        users.insert_one(
            {"username": username, "password": hashed_password, "tokens": 4}
        )
        ret_json = {
            "status": 200,
            "message": "User registered successfully",
        }

        return jsonify(ret_json)


def verify_pwd(username, password):
    """Verify user credentials."""
    if not user_exists(username=username):
        return False
    hashed_password = users.find_one({"username": username})["password"]

    if bcrypt.hashpw(password.encode("utf-8"), hashed_password) == hashed_password:
        return True
    else:
        return False


def verify_credentials(username, password):
    """Verify user credentials."""
    if not user_exists(username=username):
        return generate_return_dict(301, "User does not exist"), True

    correct_pwd = verify_pwd(username, password)
    if not correct_pwd:
        return generate_return_dict(302, "Incorrect password"), True

    return None, False


def generate_return_dict(status, message):
    """Generate a standardized return dictionary."""
    return {"status": status, "message": message}


class Classify(Resource):
    def post(self):

        posted_data = request.get_json()
        username = posted_data.get("username")
        password = posted_data.get("password")
        url = posted_data.get("url")

        # Verify user credentials
        ret_json, error = verify_credentials(username, password)

        if error:
            return jsonify(ret_json)

        # Check if the user has enough tokens
        tokens = users.find_one({"username": username})["tokens"]
        if tokens <= 0:
            return jsonify(generate_return_dict(303, "Not enough tokens"))

        if not url:
            return jsonify({"status": 400, "message": "No URL provided"})

        # Load the image from the URL
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        img = img.resize((299, 299))  # Resize to fit
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict the class of the image
        preds = pretrained_model.predict(img_array)
        results = imagenet_utils.decode_predictions(preds, top=5)

        ret_json = {}
        for pred in results[0]:
            ret_json[pred[1]] = float(pred[2] * 100)  # Convert to percentage

        # Deduct a token for the classification
        users.update_one({"username": username}, {"$inc": {"tokens": -1}})

        return jsonify(ret_json)


class Refill(Resource):
    def post(self):
        # Get posted data
        posted_data = request.get_json()
        # Get credentials
        username = posted_data.get("username")
        password = posted_data.get("admin_password")
        amount = posted_data.get("amount")
        # Check if the user exists
        if not user_exists(username):
            ret_json = generate_return_dict(301, "User does not exist")
            return jsonify(ret_json)
        correct_pwd = "abc123"  # This should be replaced with a secure admin password
        if not password == correct_pwd:
            ret_json = generate_return_dict(302, "Incorrect password")
            return jsonify(ret_json)
        # Check admin password

        # Refill tokens
        users.update_one({"username": username}, {"$set": {"tokens": amount}})
        return jsonify(generate_return_dict(200, "Tokens refilled successfully"))


api.add_resource(Register, "/register")
api.add_resource(Classify, "/classify")
api.add_resource(Refill, "/refill")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
