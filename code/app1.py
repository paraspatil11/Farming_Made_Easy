# from flask import Flask
# from dotenv import load_dotenv
# from config import Config

# # Load environment variables from .env
# load_dotenv()

# # Initialize Flask app
# app = Flask(__name__)

# # Load configuration from config.py
# app.config.from_object(Config)

# # Check the configuration by printing it
# print("Appwrite Endpoint:", app.config["APPWRITE_ENDPOINT"])

# # Define a route to avoid 404 error
# @app.route('/')
# def home():
#     return "Welcome to the Flask app!"

# if __name__ == "__main__":
#     app.run(debug=True)
