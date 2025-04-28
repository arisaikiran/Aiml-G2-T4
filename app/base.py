import firebase_admin
from firebase_admin import credentials, auth

# Initialize Firebase Admin SDK
cred = credentials.Certificate('C:\Users\Srujana\Desktop\flaskapp\venv\disease-prediction-981b1-firebase-adminsdk-vjnq7-b2f983e01a.json')
firebase_admin.initialize_app(cred)
