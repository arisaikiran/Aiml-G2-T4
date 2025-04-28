from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from dotenv import load_dotenv
load_dotenv()
import os
from PIL import Image
import numpy as np

from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import pickle
from supabase import create_client, Client

SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_URL = "https://paoowvnlwlvlnychahfh.supabase.co"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
if not SUPABASE_SERVICE_ROLE_KEY:
    print("⚠️ ERROR: Service Role Key is missing!")


app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize Firebase
ADMIN_EMAIL = "vikii91@gmail.com"
ADMIN_PASSWORD = "admin123" 

# Load models
model = pickle.load(open('C:\\Users\\Srujana\\Desktop\\flaskapp\\cnn_model.pkl', 'rb'))
model1 = load_model('C:\\Users\\Srujana\\Desktop\\flaskapp\\cnn_model_3.h5')
mri_check_model = load_model('C:\\Users\\Srujana\\Desktop\\flaskapp\\final_model_mri_vs_nonmri.h5')  # MRI classification model

import atexit
import shutil

UPLOAD_FOLDER = 'static/uploads'

# Register cleanup function to remove all files in the uploads folder when the app closes
@atexit.register
def cleanup_uploads():
    if os.path.exists(UPLOAD_FOLDER):
        print("Cleaning up uploaded files...")
        shutil.rmtree(UPLOAD_FOLDER)  # Deletes the entire folder and its contents
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Recreate the folder for future uploads

def is_admin():
    """Check if the logged-in user is an admin."""
    id = session.get('user_id')
    return id == "admin"

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/')
def home():
    
    return render_template('login_signup.html')

@app.route('/logout')
def logout():
    """Logs out the user by clearing session data."""
    cleanup_uploads()

    session.clear()
    flash("Logged out successfully!")
    return redirect(url_for('home'))

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    # Check if the user is the hardcoded admin
    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        session['user_id'] = "admin"
        session['user_email'] = email
        flash("Admin login successful!")
        return redirect(url_for('admin_dashboard'))

    try:
        # Attempt to log in with Supabase
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
         # Debugging: Print the response

        # Check if the response contains user data
        if hasattr(response, 'user') and response.user:
            session['user_id'] = response.user.id
            session['user_email'] = response.user.email
            flash("Login successful!")
            print("Session after login:", session)  # Debugging: Print session data
            
            return redirect(url_for('upload_image'))
        else:
            flash("Invalid credentials.")
    except Exception as e:
        flash(f"Login error: {str(e)}")
        print(f"Login error: {str(e)}")  # Debugging: Print the error

    return redirect(url_for('home'))

@app.route('/signup', methods=['POST'])
def signup():
    email = request.form['email']
    password = request.form['password']

    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        flash('User created successfully, please log in!')
    except Exception as e:
        flash(f"Signup error: {str(e)}")

    return redirect(url_for('home'))


@app.route('/admin')
def admin_dashboard():
    """Admin panel to view and manage users."""
    if not is_admin():
        flash("Access Denied! Admins only.")
        return redirect(url_for('home'))

    try:
        # Create a new client with service role key for admin operations
        admin_supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        
        # Fetch all users using the admin client
        response = admin_supabase.auth.admin.list_users()
        users = response if response else []
        model_accuracy = {"GAN-Augmented CNN": 95.4, "Normal CNN": 92.7}

        # Fetch recent predictions using the admin client
        predictions_response = admin_supabase.table("predictions").select("*").order("timestamp", desc=True).limit(5).execute()
        recent_predictions = predictions_response.data if hasattr(predictions_response, "data") else []

    except Exception as e:
        flash(f"Error fetching data: {str(e)}")
        users = []
        recent_predictions = []
        model_accuracy = {}

    return render_template('admin_dashboard.html', users=users, model_accuracy=model_accuracy, recent_predictions=recent_predictions)

@app.route('/admin/add_user', methods=['POST'])
def add_user():
    """Admin function to add a user."""
    if not is_admin():
        return jsonify({"error": "Unauthorized"}), 403

    email = request.form['email']
    password = request.form['password']

    try:
        # Create a new client with service role key for admin operations
        admin_supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        response = admin_supabase.auth.admin.create_user({"email": email, "password": password, "email_confirm": True})
        flash(f"User {email} added successfully!")
    except Exception as e:
        flash(f"Error adding user: {str(e)}")

    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_user/<uid>', methods=['POST'])
def delete_user(uid):
    """Admin function to delete a user."""
    if not is_admin():
        return jsonify({"error": "Unauthorized"}), 403

    try:
        # Create a new client with service role key for admin operations
        admin_supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        admin_supabase.auth.admin.delete_user(uid)
        flash("User deleted successfully!")
    except Exception as e:
        flash(f"Error deleting user: {str(e)}")

    return redirect(url_for('admin_dashboard'))

def login_required(f):
    """Decorator to ensure the user is logged in before accessing a route."""
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash("You must be logged in to access this page.")
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


@app.route('/upload', methods=['GET', 'POST'])
@login_required 
def upload_image():
    

    if request.method == 'POST':
        
        file = request.files['image']
        if file:
            # Sanitize the filename
            filename = secure_filename(file.filename)
            image_path = os.path.join('static/uploads', filename)  # Save in the static folder
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            # Save the file
            file.save(image_path)
            
            # Check if the uploaded image is an MRI using the model
            if not is_mri_image(image_path):
                flash("The uploaded image is not recognized as an MRI. Please upload a valid MRI scan.")
                return render_template('error.html', message="Invalid image uploaded! Please upload a valid MRI scan.")
            
            # Run predictions using GAN-Augmented CNN only
            label_gan, confidence_gan, message_gan = classify_with_model(image_path, model, model_name="GAN-Augmented CNN")
            
            # Pass only GAN-Augmented CNN prediction to the template
            return render_template(
                'prediction.html', 
                filepath=image_path, 
                label_gan=label_gan, confidence_gan=confidence_gan, message_gan=message_gan,
                show_normal=False  # Hide the Normal CNN prediction initially
            )
    return render_template('upload.html')




@app.route('/predict_normal', methods=['POST'])
@login_required 
def predict_normal():
    image_path = request.form['image_path']  # Get image path from the form

    # Run predictions using both models
    label_gan, confidence_gan, message_gan = classify_with_model(image_path, model, model_name="GAN-Augmented CNN")
    label_normal, confidence_normal, message_normal = classify_with_model(image_path, model1, model_name="Normal CNN")

    # Pass predictions for both models to the template
    return render_template(
        'prediction.html',
        filepath=image_path, 
        label_gan=label_gan, confidence_gan=confidence_gan, message_gan=message_gan,
        label_normal=label_normal, confidence_normal=confidence_normal, message_normal=message_normal,
        show_normal=True  # Show Normal CNN prediction
    )




def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))  # Resize to match CNN input shape
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)

def classify_with_model(image_path, selected_model, model_name):
    image_array = preprocess_image(image_path)
    
    # Prediction
    prediction = selected_model.predict(image_array)[0][0]
    label = "Disease Detected" if prediction > 0.5 else "No Disease"
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

    # Cap confidence at 99% if it equals 100%
    if confidence == 100.0:
        confidence = 98.0

    # Truncate confidence to integer
    confidence = int(confidence)
    user_email = session.get("user_email", "unknown_user")

    # Save prediction to database
    try:
        data = {
            "user_email": user_email,
            "image_path": image_path,
            "result": label,
            "confidence": confidence,
            "model_name": model_name
        }
        supabase.table("predictions").insert(data).execute()
    except Exception as e:
        print(f"Error saving prediction to database: {str(e)}")
    # Generate messages based on confidence levels
    if confidence == 99:
        if label == "Disease Detected":
            message = (
                f"{model_name}: The system has detected signs of cardiovascular disease with 99% confidence. "
                "It is strongly recommended to consult a medical professional immediately for further evaluation and treatment."
            )
        else:
            message = (
                f"{model_name}: The system has analyzed the image and found no signs of cardiovascular disease with 99% confidence. "
                "However, regular check-ups are recommended for overall health."
            )
    elif 80 <= confidence < 99:
        if label == "Disease Detected":
            message = (
                f"{model_name}: The system predicts a likelihood of cardiovascular disease with {confidence}% confidence. "
                "It is recommended to consult a healthcare professional for further evaluation."
            )
        else:
            message = (
                f"{model_name}: The system predicts no significant signs of cardiovascular disease with {confidence}% confidence. "
                "However, routine checkups are always beneficial."
            )
    elif 50 <= confidence < 80:
        message = (
            f"{model_name}: The system predicts a possibility of cardiovascular disease with {confidence}% confidence. "
            "Additional tests and medical advice are recommended to verify this result."
        )
    else:
        message = (
            f"{model_name}: The system has a low confidence ({confidence}%) in detecting cardiovascular disease from this image. "
            "However, if you have concerns, consulting a medical professional is advisable."
        )

    return label, f"{confidence}%", message


# Function to check if the uploaded image is an MRI using the MRI CNN model
def is_mri_image(image_path):
    """Check if the uploaded image is likely an MRI using a CNN model."""
    image_array = preprocess_image(image_path)  # Preprocess the image
    prediction = mri_check_model.predict(image_array)  # Predict if it's MRI
    is_mri = prediction[0] > 0.7  # If the model predicts more than 70% confidence for "MRI"
    return is_mri





if __name__ == '__main__':
    app.run(debug=True)
