# combined_app_with_admin_custom_popup_fixed.py
import os
import cv2
import base64
import sqlite3
import numpy as np
import face_recognition
from PIL import Image
from io import BytesIO
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, Response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import requests # Keep for Pollinations API call
from rembg import remove
from datetime import datetime
import mediapipe as mp
from deepface import DeepFace
from jinja2 import DictLoader
import logging
import time # Added missing import
import urllib.parse # For URL encoding
import json # For handling potential JSON responses
import functools # Added missing import for login_required decorator

# Configure logging
logging.basicConfig(level=logging.INFO) # Changed to INFO to see filter download messages
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-change-this-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# --- Filter Image URLs ---
FILTER_IMAGE_URLS = {
    'glasses': 'https://static.vecteezy.com/system/resources/thumbnails/046/158/728/small_2x/black-eyeglasses-frame-png.png',
    'mustache': 'https://png.pngtree.com/png-vector/20240628/ourmid/pngtree-hercule-poirot-fake-moustache-isolated-png-image_12722115.png',
    'hat': 'https://u.cubeupload.com/mcgdj/hat.png',
    'dog_nose': 'https://www.citypng.com/public/uploads/preview/hd-snapchat-cute-dalmatian-dog-puppy-filter-png-image-704081694687639gtqldjovft.png'
}

# --- Admin Check Helper ---
def is_admin():
    return 'user_id' in session and session.get('username') == 'Desmond'

# --- Login Required Decorator ---
def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/processed', exist_ok=True)
os.makedirs('static/filters', exist_ok=True)

# --- Function to Download Filter Images ---
def download_filter_images():
    """Downloads filter images if they are missing."""
    for filter_name, url in FILTER_IMAGE_URLS.items():
        filename = f"{filter_name}.png"
        filepath = os.path.join('static/filters', filename)
        if not os.path.exists(filepath):
            logger.info(f"Filter image '{filename}' not found. Attempting to download from {url}...")
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Successfully downloaded '{filename}'.")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download '{filename}' from {url}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error downloading '{filename}': {e}")
        else:
            logger.info(f"Filter image '{filename}' already exists.")

# --- Load filter images with validation ---
def load_filter_images():
    """Loads filter images into memory."""
    filter_images = {}
    for filter_name in ['glasses', 'mustache', 'hat', 'dog_nose']:
        img_path = f'static/filters/{filter_name}.png'
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                filter_images[filter_name] = img
                logger.info(f"Loaded filter image '{filter_name}'.")
            else:
                logger.warning(f"Filter image '{filter_name}' failed to load with cv2.")
                filter_images[filter_name] = None
        else:
            logger.warning(f"Filter image file '{img_path}' not found.")
            filter_images[filter_name] = None
    return filter_images

# --- Database setup (Updated with admin_ips table) ---
def init_db():
    conn = sqlite3.connect('face_app.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    # --- Admin Feature: IP Banning Table ---
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS banned_ips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT UNIQUE NOT NULL,
            banned_by INTEGER,
            reason TEXT,
            banned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (banned_by) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

# --- Initialize DB and Filters ---
init_db()
download_filter_images() # Download missing filters on startup
filter_images = load_filter_images() # Load filters into memory

# --- IP Banning Middleware (Conceptual - Add to relevant routes or globally) ---
# @app.before_request
# def block_banned_ips():
#     user_ip = request.environ.get('REMOTE_ADDR')
#     conn = sqlite3.connect('face_app.db')
#     cursor = conn.cursor()
#     cursor.execute('SELECT 1 FROM banned_ips WHERE ip_address = ?', (user_ip,))
#     if cursor.fetchone():
#         conn.close()
#         return "Access Denied: Your IP has been banned.", 403
#     conn.close()

# Face recognition utilities
class FaceRecognizer:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.load_known_faces()
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh

    def load_known_faces(self):
        conn = sqlite3.connect('face_app.db')
        cursor = conn.cursor()
        cursor.execute('SELECT name, encoding FROM face_encodings')
        results = cursor.fetchall()
        self.known_names = []
        self.known_encodings = []
        for name, encoding_blob in results:
            encoding = np.frombuffer(encoding_blob, dtype=np.float64)
            self.known_encodings.append(encoding)
            self.known_names.append(name)
        conn.close()

    def add_face(self, user_id, name, image_path):
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            return False, "No face detected in the image"
        encoding = encodings[0]
        conn = sqlite3.connect('face_app.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO face_encodings (user_id, name, encoding, image_path)
            VALUES (?, ?, ?, ?)
        ''', (user_id, name, encoding.tobytes(), image_path))
        conn.commit()
        conn.close()
        self.known_encodings.append(encoding)
        self.known_names.append(name)
        return True, "Face added successfully"

    def recognize_face(self, image_path):
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        results = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_names[first_match_index]
            results.append(name)
        return results, face_locations

    def detect_emotion(self, image_path):
        try:
            result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
            if result and isinstance(result, list) and len(result) > 0:
                return result[0]['dominant_emotion']
            return "Unknown"
        except Exception as e:
            app.logger.error(f"Error in detect_emotion: {str(e)}")
            return "Unknown"

    def log_attendance(self, user_id, name):
        conn = sqlite3.connect('face_app.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO attendance (user_id, name)
            VALUES (?, ?)
        ''', (user_id, name))
        conn.commit()
        conn.close()

face_recognizer = FaceRecognizer()

# --- Helper function for face swapping ---
def warp_im(img, M, dshape):
    """Apply affine transformation to align face."""
    output_im = np.zeros(dshape, dtype=img.dtype)
    cv2.warpAffine(img, M[:2], (dshape[1], dshape[0]), dst=output_im, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    """Correct color balance between swapped face and target face."""
    blur_amount = 0.6 * np.linalg.norm(
                              np.mean(landmarks1[36:42], axis=0) -
                              np.mean(landmarks1[42:48], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))

def face_swap_core(img1, img2, landmarks1, landmarks2):
    """Core face swapping logic."""
    # Calculate transformation matrices
    M1 = cv2.estimateAffinePartial2D(landmarks1, landmarks2, method=cv2.LMEDS)[0] # Source to Target
    M2 = cv2.estimateAffinePartial2D(landmarks2, landmarks1, method=cv2.LMEDS)[0] # Target to Source

    if M1 is None or M2 is None:
        raise ValueError("Could not estimate affine transform.")

    # Warp images
    warped_img1 = warp_im(img1, M1, img2.shape) # Source warped to target shape
    warped_img2 = warp_im(img2, M2, img1.shape) # Target warped to source shape

    # Create masks for faces
    mask1 = np.zeros_like(img1, dtype=np.uint8)
    hull1 = cv2.convexHull(landmarks1.astype(np.int32))
    cv2.fillConvexPoly(mask1, hull1, (255, 255, 255))

    mask2 = np.zeros_like(img2, dtype=np.uint8)
    hull2 = cv2.convexHull(landmarks2.astype(np.int32))
    cv2.fillConvexPoly(mask2, hull2, (255, 255, 255))

    # Warp masks
    warped_mask1 = warp_im(mask1, M1, img2.shape)
    warped_mask2 = warp_im(mask2, M2, img1.shape)

    # Combine images and masks for blending
    combined_mask = np.maximum(warped_mask1, warped_mask2) # Use the larger mask

    # Color correction
    warped_img1_corrected = correct_colours(img2, warped_img1, landmarks2) # Correct warped source to match target
    warped_img2_corrected = correct_colours(img1, warped_img2, landmarks1) # Correct warped target to match source

    # Blend
    output_im = img2 * (1.0 - combined_mask / 255.0) + warped_img1_corrected * (combined_mask / 255.0)
    output_im = np.clip(output_im, 0, 255).astype(np.uint8) # Ensure valid pixel values

    return output_im

# --- Routes ---
@app.route('/')
@login_required
def index():
    return render_template('index.html', is_admin=is_admin())

@app.route('/live_camera')
@login_required
def live_camera():
    return render_template('live_camera.html')

# --- Admin Routes ---
@app.route('/admin')
@login_required
def admin_panel():
    if not is_admin():
        flash("Access denied.", "danger")
        return redirect(url_for('index'))
    conn = sqlite3.connect('face_app.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT fe.id, fe.name, fe.created_at, u.username
        FROM face_encodings fe
        JOIN users u ON fe.user_id = u.id
        ORDER BY fe.created_at DESC
    ''')
    faces = cursor.fetchall()

    cursor.execute('''
        SELECT id, ip_address, reason, banned_at
        FROM banned_ips
        ORDER BY banned_at DESC
    ''')
    banned_ips = cursor.fetchall()
    conn.close()
    return render_template('admin.html', faces=faces, banned_ips=banned_ips)

@app.route('/admin/delete_face/<int:face_id>', methods=['POST'])
@login_required
def delete_face(face_id):
    if not is_admin():
        return jsonify({'error': 'Access denied.'}), 403
    try:
        conn = sqlite3.connect('face_app.db')
        cursor = conn.cursor()
        # Optional: Delete the associated image file first
        cursor.execute('SELECT image_path FROM face_encodings WHERE id = ?', (face_id,))
        result = cursor.fetchone()
        if result and result[0] and os.path.exists(result[0]):
            os.remove(result[0])

        cursor.execute('DELETE FROM face_encodings WHERE id = ?', (face_id,))
        conn.commit()
        conn.close()
        face_recognizer.load_known_faces() # Reload known faces
        # Return success message for frontend handling
        return jsonify({'message': 'Face deleted successfully.'})
    except Exception as e:
        app.logger.error(f"Error deleting face {face_id}: {e}")
        return jsonify({'error': 'Failed to delete face.'}), 500

@app.route('/admin/ban_ip', methods=['POST'])
@login_required
def ban_ip():
    if not is_admin():
        return jsonify({'error': 'Access denied.'}), 403
    ip_address = request.form.get('ip_address')
    reason = request.form.get('reason', 'No reason provided')
    if not ip_address:
        return jsonify({'error': 'IP address is required.'}), 400
    try:
        conn = sqlite3.connect('face_app.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO banned_ips (ip_address, banned_by, reason)
            VALUES (?, ?, ?)
        ''', (ip_address, session['user_id'], reason))
        conn.commit()
        conn.close()
        # Return success message for frontend handling
        return jsonify({'message': f'IP {ip_address} banned successfully.'})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'IP address is already banned.'}), 400
    except Exception as e:
        app.logger.error(f"Error banning IP {ip_address}: {e}")
        return jsonify({'error': 'Failed to ban IP.'}), 500

@app.route('/admin/unban_ip/<int:ban_id>', methods=['POST'])
@login_required
def unban_ip(ban_id):
    if not is_admin():
        return jsonify({'error': 'Access denied.'}), 403
    try:
        conn = sqlite3.connect('face_app.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM banned_ips WHERE id = ?', (ban_id,))
        conn.commit()
        conn.close()
        # Return success message for frontend handling
        return jsonify({'message': 'IP unbanned successfully.'})
    except Exception as e:
        app.logger.error(f"Error unbanning IP with ID {ban_id}: {e}")
        return jsonify({'error': 'Failed to unban IP.'}), 500


def gen_frames():
    camera = cv2.VideoCapture(0)
    try:
        recognizer = face_recognizer
        while True:
            success, frame = camera.read()
            if not success:
                break
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(recognizer.known_encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = recognizer.known_names[first_match_index]
                face_names.append(name)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                color = (0, 0, 255) if name != "Unknown" else (255, 0, 0)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if not username or not email or not password:
            flash('All fields are required', 'danger')
            return render_template('register.html')
        conn = sqlite3.connect('face_app.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        if cursor.fetchone():
            flash('Username or email already exists', 'warning')
            conn.close()
            return render_template('register.html')
        password_hash = generate_password_hash(password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
        ''', (username, email, password_hash))
        conn.commit()
        conn.close()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # --- IP Banning Check (Conceptual) ---
    # user_ip = request.environ.get('REMOTE_ADDR')
    # conn = sqlite3.connect('face_app.db')
    # cursor = conn.cursor()
    # cursor.execute('SELECT 1 FROM banned_ips WHERE ip_address = ?', (user_ip,))
    # if cursor.fetchone():
    #     conn.close()
    #     flash("Access Denied: Your IP has been banned.", "danger")
    #     return render_template('login.html')
    # conn.close()

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('face_app.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# --- Updated Backend Route Logic ---
# Priority: Check for image_data (base64) first, then fall back to file upload.

@app.route('/add_face', methods=['POST'])
@login_required
def add_face():
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Name is required'}), 400

    image_data = None
    temp_filename = None

    # Handle base64 data from live camera capture (Priority 1)
    if 'image_data' in request.form:
        image_data = request.form['image_data']
        try:
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            # Save temporarily
            temp_filename = secure_filename(f"{session['user_id']}_{name}_{int(time.time())}_capture.jpg")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            cv2.imwrite(filepath, image)
        except Exception as e:
            app.logger.error(f"Error processing live camera image for add_face: {str(e)}")
            return jsonify({'error': 'Failed to process captured image'}), 500
    # Handle file upload (Priority 2)
    elif 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not file.content_type.startswith('image/'):
            return jsonify({'error': 'Invalid file type'}), 400
        temp_filename = secure_filename(f"{session['user_id']}_{name}_{int(time.time())}_upload.jpg")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        try:
             file.save(filepath)
        except Exception as e:
             app.logger.error(f"Error saving uploaded file for add_face: {str(e)}")
             return jsonify({'error': 'Failed to save uploaded image'}), 500
    else:
        return jsonify({'error': 'No image provided (file or capture)'}), 400

    if not temp_filename or not os.path.exists(filepath):
        return jsonify({'error': 'Could not process image'}), 500

    try:
        success, message = face_recognizer.add_face(session['user_id'], name, filepath)
        if success:
            # Optional: Rename/move the file to a permanent location if needed
            # For now, we keep it in uploads and it's referenced in the DB
            # The DB stores the filepath, so we don't delete it here.
            return jsonify({'message': message})
        else:
            # Remove the file if face wasn't added
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': message}), 400
    except Exception as e:
        app.logger.error(f"Error in add_face: {str(e)}")
        # Ensure file is removed on any error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': 'An error occurred while adding the face'}), 500


@app.route('/recognize_face', methods=['POST'])
@login_required
def recognize_face():
    image_data = None
    temp_filename = None
    # Handle base64 data from live camera capture (Priority 1)
    if 'image_data' in request.form:
        image_data = request.form['image_data']
        try:
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            temp_filename = secure_filename(f"temp_{session['user_id']}_{int(time.time())}_capture.jpg")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            cv2.imwrite(filepath, image)
        except Exception as e:
            app.logger.error(f"Error processing live camera image for recognize_face: {str(e)}")
            return jsonify({'error': 'Failed to process captured image'}), 500
    # Handle file upload (Priority 2)
    elif 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not file.content_type.startswith('image/'):
            return jsonify({'error': 'Invalid file type'}), 400
        temp_filename = secure_filename(f"temp_{session['user_id']}_{int(time.time())}_upload.jpg")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        try:
            file.save(filepath)
        except Exception as e:
            app.logger.error(f"Error saving uploaded file for recognize_face: {str(e)}")
            return jsonify({'error': 'Failed to save uploaded image'}), 500
    else:
       return jsonify({'error': 'No image provided (file or capture)'}), 400

    if not temp_filename or not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)):
        return jsonify({'error': 'Could not process image'}), 500

    try:
        names, locations = face_recognizer.recognize_face(filepath)
        emotion = face_recognizer.detect_emotion(filepath)
        for name in names:
            if name != "Unknown":
                face_recognizer.log_attendance(session['user_id'], name)
        image = cv2.imread(filepath)
        for (top, right, bottom, left), name in zip(locations, names):
            color = (0, 0, 255) if name != "Unknown" else (255, 0, 0)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            cv2.putText(image, name, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            if emotion != "Unknown":
                cv2.putText(image, f"Emotion: {emotion}", (left, top - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        annotated_filename = f"recognized_{temp_filename}"
        annotated_path = os.path.join('static/processed', annotated_filename)
        cv2.imwrite(annotated_path, image)
        return jsonify({
            'names': names,
            'emotion': emotion,
            'annotated_image': f'/static/processed/{annotated_filename}'
        })
    except Exception as e:
        app.logger.error(f"Error in recognize_face: {str(e)}")
        return jsonify({'error': 'An error occurred during face recognition'}), 500
    finally:
        # Ensure temporary file is always removed
        if temp_filename and os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], temp_filename))


@app.route('/roast_image', methods=['POST'])
@login_required
def roast_image():
    image_data = None
    temp_filename = None

    # Handle base64 data from live camera capture (Priority 1)
    if 'image_data' in request.form:
        image_data = request.form['image_data'].split(',')[1] # Get base64 part
    # Handle file upload (Priority 2)
    elif 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        if not file.content_type.startswith('image/'):
            return jsonify({'error': 'Invalid file type'}), 400
        temp_filename = secure_filename(f"roast_{session['user_id']}_{int(time.time())}_upload.jpg")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        try:
            file.save(filepath)
            with open(filepath, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            app.logger.error(f"Error reading uploaded file for roasting: {str(e)}")
            return jsonify({'error': 'Failed to process uploaded image'}), 500
    else:
        return jsonify({'error': 'No image provided (file or capture)'}), 400

    if not image_data:
        if temp_filename and os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)):
             os.remove(os.path.join(app.config['UPLOAD_FOLDER'], temp_filename))
        return jsonify({'error': 'Could not process image data'}), 500

    # Create data URL for the image
    image_data_url = f"image/jpeg;base64,{image_data}" # Correct format for Pollinations

    # --- Use text.pollinations.ai API ---
    # Construct the prompt. Including the image data URL directly in the prompt string
    # is a common way for such APIs to handle images if they support vision.
    prompt = f"Roast this person in the image. Be creative and funny but not mean-spirited. Keep it short. Image: {image_data_url}"
    encoded_prompt = urllib.parse.quote(prompt, safe='') # URL encode the entire prompt
    pollinations_url = f"https://text.pollinations.ai/{encoded_prompt}?model=openai"

    try:
        # Make GET request
        response = requests.get(pollinations_url, timeout=30) # Add a timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        roast_text = response.text.strip() # Get the response text

        if not roast_text:
            roast_text = "Hmm, I couldn't come up with a roast this time. Maybe try another image?"

        return jsonify({'roast': roast_text})
    except requests.exceptions.Timeout:
        app.logger.error("Timeout error connecting to Pollinations API")
        return jsonify({'error': 'The roasting service took too long to respond. Please try again.'}), 504 # Gateway Timeout
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error connecting to Pollinations API: {str(e)}")
        return jsonify({'error': 'Failed to connect to the roasting service'}), 502 # Bad Gateway
    except Exception as e:
        app.logger.error(f"Unexpected error in roast_image: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred while roasting the image'}), 500
    finally:
        # Ensure temporary file is always removed if it was created from an upload
        if temp_filename and os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], temp_filename))


@app.route('/remove_background', methods=['POST'])
@login_required
def remove_background():
    image_data = None
    temp_filename = None
    # Handle base64 data from live camera capture (Priority 1)
    if 'image_data' in request.form:
        image_data = request.form['image_data']
        try:
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            pil_image = Image.fromarray(cv2.cvtColor(cv2.imdecode(image_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        except Exception as e:
            app.logger.error(f"Error processing live camera image for remove_background: {str(e)}")
            return jsonify({'error': 'Failed to process captured image'}), 500
    # Handle file upload (Priority 2)
    elif 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not file.content_type.startswith('image/'):
            return jsonify({'error': 'Invalid file type'}), 400
        try:
            pil_image = Image.open(file.stream)
        except Exception as e:
            app.logger.error(f"Error opening uploaded file for remove_background: {str(e)}")
            return jsonify({'error': 'Failed to open uploaded image'}), 500
    else:
       return jsonify({'error': 'No image provided (file or capture)'}), 400

    try:
        # Process the PIL Image
        output_image = remove(pil_image)
        filename = secure_filename(f"bg_remove_{session['user_id']}_{int(time.time())}.png")
        output_path = os.path.join('static/processed', filename)
        output_image.save(output_path)
        return jsonify({
            'processed_image': f'/static/processed/{filename}'
        })
    except Exception as e:
        app.logger.error(f"Error in remove_background: {str(e)}")
        return jsonify({'error': 'An error occurred during background removal'}), 500


@app.route('/apply_filter', methods=['POST'])
@login_required
def apply_filter():
    # Reload filter images in case they were updated
    global filter_images
    filter_images = load_filter_images()

    image_data = None
    temp_filename = None
    filter_type = request.form.get('filter_type', 'glasses')
    # Handle base64 data from live camera capture (Priority 1)
    if 'image_data' in request.form:
        image_data = request.form['image_data']
        try:
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            temp_filename = secure_filename(f"filtered_{session['user_id']}_{int(time.time())}_capture.png")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            cv2.imwrite(filepath, image_bgr) # Save BGR for OpenCV processing later
        except Exception as e:
            app.logger.error(f"Error processing live camera image for apply_filter: {str(e)}")
            return jsonify({'error': 'Failed to process captured image'}), 500
    # Handle file upload (Priority 2)
    elif 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        filter_type = request.form.get('filter_type', 'glasses') # Get filter type from form data
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not file.content_type.startswith('image/'):
            return jsonify({'error': 'Invalid file type'}), 400
        if filter_type not in filter_images or filter_images[filter_type] is None:
            return jsonify({'error': f'Filter {filter_type} not available or failed to load'}), 400
        temp_filename = secure_filename(f"filtered_{session['user_id']}_{int(time.time())}_upload.png")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        try:
           file.save(filepath) # Save file first
           image_bgr = cv2.imread(filepath) # Then read it with OpenCV
           if image_bgr is None:
                raise ValueError("Could not read image file.")
           image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
             app.logger.error(f"Error saving/reading uploaded file for apply_filter: {str(e)}")
             if os.path.exists(filepath):
                 os.remove(filepath)
             return jsonify({'error': 'Failed to process uploaded image'}), 500
    else:
       return jsonify({'error': 'No image provided (file or capture)'}), 400

    if filter_type not in filter_images or filter_images[filter_type] is None:
        if temp_filename and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Filter {filter_type} not available or failed to load'}), 400

    try:
        # Apply filter logic (using image_rgb and image_bgr)
        img = image_bgr # Use BGR image for OpenCV operations
        rgb_img = image_rgb # Use RGB image for MediaPipe
        with face_recognizer.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(rgb_img)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    if filter_type == 'glasses' and filter_images['glasses'] is not None:
                        nose_tip = face_landmarks.landmark[4]
                        h, w, _ = img.shape
                        x = int(nose_tip.x * w) - 50
                        y = int(nose_tip.y * h) - 50
                        glasses = filter_images['glasses']
                        glasses = cv2.resize(glasses, (100, 50))
                        x1, y1 = max(x, 0), max(y, 0)
                        x2, y2 = min(x+100, w), min(y+50, h)
                        gw, gh = x2-x1, y2-y1
                        if gw > 0 and gh > 0:
                            for c in range(0, 3):
                                img[y1:y1+gh, x1:x1+gw, c] = (
                                    glasses[0:gh, 0:gw, c] * (glasses[0:gh, 0:gw, 3] / 255.0) +
                                    img[y1:y1+gh, x1:x1+gw, c] * (1.0 - glasses[0:gh, 0:gw, 3] / 255.0)
                                )
                    elif filter_type == 'mustache' and filter_images['mustache'] is not None:
                        nose_tip = face_landmarks.landmark[4]
                        h, w, _ = img.shape
                        x = int(nose_tip.x * w) - 30
                        y = int(nose_tip.y * h) + 10
                        mustache = filter_images['mustache']
                        mustache = cv2.resize(mustache, (60, 30))
                        x1, y1 = max(x, 0), max(y, 0)
                        x2, y2 = min(x+60, w), min(y+30, h)
                        gw, gh = x2-x1, y2-y1
                        if gw > 0 and gh > 0:
                            for c in range(0, 3):
                                img[y1:y1+gh, x1:x1+gw, c] = (
                                    mustache[0:gh, 0:gw, c] * (mustache[0:gh, 0:gw, 3] / 255.0) +
                                    img[y1:y1+gh, x1:x1+gw, c] * (1.0 - mustache[0:gh, 0:gw, 3] / 255.0)
                                )
                    elif filter_type == 'hat' and filter_images['hat'] is not None:
                        nose_tip = face_landmarks.landmark[4]
                        h, w, _ = img.shape
                        x = int(nose_tip.x * w) - 50
                        y = int(nose_tip.y * h) - 80
                        hat = filter_images['hat']
                        hat = cv2.resize(hat, (100, 50))
                        x1, y1 = max(x, 0), max(y, 0)
                        x2, y2 = min(x+100, w), min(y+50, h)
                        gw, gh = x2-x1, y2-y1
                        if gw > 0 and gh > 0:
                            for c in range(0, 3):
                                img[y1:y1+gh, x1:x1+gw, c] = (
                                    hat[0:gh, 0:gw, c] * (hat[0:gh, 0:gw, 3] / 255.0) +
                                    img[y1:y1+gh, x1:x1+gw, c] * (1.0 - hat[0:gh, 0:gw, 3] / 255.0)
                                )
                    elif filter_type == 'dog_nose' and filter_images['dog_nose'] is not None:
                        nose_tip = face_landmarks.landmark[4]
                        h, w, _ = img.shape
                        x = int(nose_tip.x * w) - 20
                        y = int(nose_tip.y * h) - 10
                        dog_nose = filter_images['dog_nose']
                        dog_nose = cv2.resize(dog_nose, (40, 40))
                        x1, y1 = max(x, 0), max(y, 0)
                        x2, y2 = min(x+40, w), min(y+40, h)
                        gw, gh = x2-x1, y2-y1
                        if gw > 0 and gh > 0:
                            for c in range(0, 3):
                                img[y1:y1+gh, x1:x1+gw, c] = (
                                    dog_nose[0:gh, 0:gw, c] * (dog_nose[0:gh, 0:gw, 3] / 255.0) +
                                    img[y1:y1+gh, x1:x1+gw, c] * (1.0 - dog_nose[0:gh, 0:gw, 3] / 255.0)
                                )
        filtered_path = os.path.join('static/processed', temp_filename) # Use the temp filename
        cv2.imwrite(filtered_path, img)
        return jsonify({
            'filtered_image': f'/static/processed/{temp_filename}' # Return the temp filename
        })
    except Exception as e:
        app.logger.error(f"Error in apply_filter: {str(e)}")
        return jsonify({'error': 'An error occurred while applying filter'}), 500
    finally:
        # Ensure temporary file is always removed if it was created from an upload/capture
        # Note: We are now returning the temp file as the processed image, so don't delete it here in finally.
        # The temp filename is effectively the final filename in static/processed.
        pass # Do nothing in finally for filter


@app.route('/face_swap', methods=['POST'])
@login_required
def face_swap():
    # Face swap requires two images, so live camera capture needs special handling.
    # We'll modify the frontend to capture two frames and send them.
    image_data1 = None
    image_data2 = None
    filepath1 = None
    filepath2 = None
    temp_filename1 = None
    temp_filename2 = None

    # Check for base64 data for both images (live camera capture) (Priority 1)
    if 'image_data1' in request.form and 'image_data2' in request.form:
         try:
             # Process image_data1
             header1, encoded1 = request.form['image_data1'].split(',', 1)
             image_bytes1 = base64.b64decode(encoded1)
             image_array1 = np.frombuffer(image_bytes1, dtype=np.uint8)
             img1_bgr = cv2.imdecode(image_array1, cv2.IMREAD_COLOR)
             img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
             # Save temporarily
             temp_filename1 = secure_filename(f"swap1_{session['user_id']}_{int(time.time())}_capture.jpg")
             filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename1)
             cv2.imwrite(filepath1, img1_bgr)

             # Process image_data2
             header2, encoded2 = request.form['image_data2'].split(',', 1)
             image_bytes2 = base64.b64decode(encoded2)
             image_array2 = np.frombuffer(image_bytes2, dtype=np.uint8)
             img2_bgr = cv2.imdecode(image_array2, cv2.IMREAD_COLOR)
             img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
             # Save temporarily
             temp_filename2 = secure_filename(f"swap2_{session['user_id']}_{int(time.time())}_capture.jpg")
             filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename2)
             cv2.imwrite(filepath2, img2_bgr)

         except Exception as e:
             app.logger.error(f"Error processing live camera images for face_swap: {str(e)}")
             # Cleanup potential first file
             if filepath1 and os.path.exists(filepath1): os.remove(filepath1)
             return jsonify({'error': 'Failed to process captured images'}), 500
    # Handle file uploads (Priority 2)
    elif 'image1' in request.files and 'image2' in request.files:
        file1 = request.files['image1']
        file2 = request.files['image2']
        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not (file1.content_type.startswith('image/') and file2.content_type.startswith('image/')):
            return jsonify({'error': 'Invalid file type'}), 400

        # Generate unique filenames
        temp_filename1 = secure_filename(f"swap1_{session['user_id']}_{int(time.time())}_upload.jpg")
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename1)
        temp_filename2 = secure_filename(f"swap2_{session['user_id']}_{int(time.time())}_upload.jpg")
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename2)

        try:
            # Save uploaded files
            file1.save(filepath1)
            file2.save(filepath2)

            # Load images (face_recognition works with RGB, OpenCV with BGR)
            img1_rgb = face_recognition.load_image_file(filepath1) # RGB format
            img2_rgb = face_recognition.load_image_file(filepath2) # RGB format

            # Convert to BGR for OpenCV operations
            img1_bgr = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
            img2_bgr = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
             app.logger.error(f"Error saving/loading uploaded files for face_swap: {str(e)}")
             # Cleanup
             if os.path.exists(filepath1): os.remove(filepath1)
             if os.path.exists(filepath2): os.remove(filepath2)
             return jsonify({'error': 'Failed to process uploaded images'}), 500
    else:
        return jsonify({'error': 'Two images required (files or captures)'}), 400

    # Core processing logic (same for both capture and upload)
    try:
        # Detect faces and landmarks (using RGB images for face_recognition)
        face_locations1 = face_recognition.face_locations(img1_rgb)
        face_locations2 = face_recognition.face_locations(img2_rgb)

        if not face_locations1 or not face_locations2:
            return jsonify({'error': 'Face not detected in one or both images'}), 400

        # Assume one face per image for simplicity
        face_landmarks1_list = face_recognition.face_landmarks(img1_rgb, face_locations=face_locations1)
        face_landmarks2_list = face_recognition.face_landmarks(img2_rgb, face_locations=face_locations2)

        if not face_landmarks1_list or not face_landmarks2_list:
             return jsonify({'error': 'Could not find facial landmarks in one or both images'}), 400

        # Get landmarks for the first detected face in each image
        landmarks1_dict = face_landmarks1_list[0]
        landmarks2_dict = face_landmarks2_list[0]

        # Select key landmarks for alignment (e.g., eyes, nose, mouth corners)
        # Convert landmarks to numpy arrays (y, x format for OpenCV)
        landmarks1 = np.array([(p.y, p.x) for p in [landmarks1_dict['left_eye'][0],
                                                    landmarks1_dict['left_eye'][3],
                                                    landmarks1_dict['right_eye'][0],
                                                    landmarks1_dict['right_eye'][3],
                                                    landmarks1_dict['nose_bridge'][0],
                                                    landmarks1_dict['nose_bridge'][len(landmarks1_dict['nose_bridge'])-1],
                                                    landmarks1_dict['top_lip'][0],
                                                    landmarks1_dict['top_lip'][len(landmarks1_dict['top_lip'])//2]
                                                    ]])
        landmarks2 = np.array([(p.y, p.x) for p in [landmarks2_dict['left_eye'][0],
                                                    landmarks2_dict['left_eye'][3],
                                                    landmarks2_dict['right_eye'][0],
                                                    landmarks2_dict['right_eye'][3],
                                                    landmarks2_dict['nose_bridge'][0],
                                                    landmarks2_dict['nose_bridge'][len(landmarks2_dict['nose_bridge'])-1],
                                                    landmarks2_dict['top_lip'][0],
                                                    landmarks2_dict['top_lip'][len(landmarks2_dict['top_lip'])//2]
                                                    ]])

        # Perform face swap (using BGR images for OpenCV operations)
        swapped_img_bgr = face_swap_core(img1_bgr, img2_bgr, landmarks1, landmarks2)

        # Save the result (cv2.imwrite expects BGR)
        output_filename = f"swapped_{session['user_id']}_{int(time.time())}.jpg"
        output_path = os.path.join('static/processed', output_filename)
        cv2.imwrite(output_path, swapped_img_bgr)

        return jsonify({
            'swapped_image': f'/static/processed/{output_filename}'
        })

    except ValueError as ve:
        app.logger.error(f"ValueError in face_swap: {str(ve)}")
        return jsonify({'error': str(ve)}), 400 # Return specific error message
    except Exception as e:
        app.logger.error(f"Error in face_swap: {str(e)}")
        return jsonify({'error': 'An error occurred during face swapping'}), 500
    finally:
        # Ensure temporary files are always removed (check if they exist)
        if filepath1 and os.path.exists(filepath1):
            os.remove(filepath1)
        if filepath2 and os.path.exists(filepath2):
            os.remove(filepath2)


# Define all HTML templates in a dictionary
templates_dict = {
    'base.html': '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Face Recognition App</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            body { padding-top: 60px; background-color: #f8f9fa; }
            .feature-card { transition: transform 0.3s; height: 100%; }
            .feature-card:hover { transform: translateY(-5px); }
            .camera-container { position: relative; max-width: 100%; /* Full width */ margin: 0 auto; height: 70vh; /* 70% of viewport height */ }
            #videoFeed { width: 100%; height: 100%; object-fit: cover; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
            .result-container { margin-top: 30px; }
            .result-image { max-width: 100%; border-radius: 8px; }
            .feature-icon { font-size: 2.5rem; margin-bottom: 15px; color: #0d6efd; }
            .nav-tabs .nav-link.active { font-weight: 600; }
            /* Smart Streaming Simulation */
            .processing-overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.5);
                display: none; /* Hidden by default */
                justify-content: center;
                align-items: center;
                color: white;
                z-index: 10;
                border-radius: 8px;
            }
            .scanner-line {
                 position: absolute;
                 top: 0;
                 left: 0;
                 width: 100%;
                 height: 5px;
                 background-color: #00ff00; /* Green line */
                 box-shadow: 0 0 10px #00ff00;
                 animation: scan 1s linear infinite;
                 display: none; /* Hidden by default */
             }
             @keyframes scan {
                 0% { top: 0; }
                 100% { top: 100%; }
             }
             /* Live Camera Tool Selection */
             #liveCameraTools {
                 position: absolute;
                 bottom: 20px;
                 left: 50%;
                 transform: translateX(-50%);
                 z-index: 11;
                 background-color: rgba(255, 255, 255, 0.8);
                 padding: 10px;
                 border-radius: 5px;
                 display: none; /* Hidden by default */
             }
             #liveCameraTools button {
                 margin: 0 5px;
             }
             /* Capture Button Styling */
             .capture-btn {
                 position: absolute;
                 bottom: 20px;
                 left: 50%;
                 transform: translateX(-50%);
                 z-index: 11;
                 display: none; /* Hidden by default */
             }
             /* Admin Link */
             .admin-link {
                 position: fixed;
                 top: 70px;
                 right: 20px;
                 z-index: 1000;
             }
             /* Custom Popup Styles */
             .custom-popup-overlay {
                 position: fixed;
                 top: 0;
                 left: 0;
                 width: 100%;
                 height: 100%;
                 background-color: rgba(0, 0, 0, 0.5);
                 display: none; /* Hidden by default */
                 justify-content: center;
                 align-items: center;
                 z-index: 2000; /* Higher than navbar and toasts */
             }
             .custom-popup {
                 background-color: white;
                 border-radius: 8px;
                 box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                 max-width: 500px;
                 width: 90%;
                 max-height: 80vh;
                 overflow-y: auto;
                 animation: popupFadeIn 0.3s ease-out;
             }
             @keyframes popupFadeIn {
                 from { opacity: 0; transform: translateY(-20px); }
                 to { opacity: 1; transform: translateY(0); }
             }
             .custom-popup-header {
                 padding: 15px;
                 border-bottom: 1px solid #eee;
                 display: flex;
                 justify-content: space-between;
                 align-items: center;
             }
             .custom-popup-title {
                 margin: 0;
                 font-size: 1.25rem;
                 font-weight: 500;
             }
             .custom-popup-close {
                 background: none;
                 border: none;
                 font-size: 1.5rem;
                 cursor: pointer;
                 color: #aaa;
                 padding: 0;
                 width: 30px;
                 height: 30px;
                 display: flex;
                 align-items: center;
                 justify-content: center;
                 border-radius: 50%;
             }
             .custom-popup-close:hover {
                 color: #000;
                 background-color: #f0f0f0;
             }
             .custom-popup-body {
                 padding: 20px;
             }
             .custom-popup-footer {
                 padding: 15px;
                 border-top: 1px solid #eee;
                 display: flex;
                 justify-content: flex-end;
                 gap: 10px;
             }
             .custom-popup-btn {
                 padding: 8px 16px;
                 border-radius: 4px;
                 border: 1px solid transparent;
                 cursor: pointer;
                 font-size: 0.875rem;
             }
             .custom-popup-btn-primary {
                 background-color: #0d6efd;
                 color: white;
                 border-color: #0d6efd;
             }
             .custom-popup-btn-primary:hover {
                 background-color: #0b5ed7;
                 border-color: #0a58ca;
             }
             .custom-popup-btn-secondary {
                 background-color: #6c757d;
                 color: white;
                 border-color: #6c757d;
             }
             .custom-popup-btn-secondary:hover {
                 background-color: #5c636a;
                 border-color: #565e64;
             }
             .custom-popup-message {
                 margin: 0;
                 word-wrap: break-word; /* Handle long messages */
             }
             .custom-popup-message.error {
                 color: #dc3545; /* Bootstrap danger color */
             }
             .custom-popup-message.success {
                 color: #198754; /* Bootstrap success color */
             }
             .custom-popup-message.warning {
                 color: #ffc107; /* Bootstrap warning color */
             }
             .custom-popup-message.info {
                 color: #0d6efd; /* Bootstrap primary color */
             }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
            <div class="container">
                <a class="navbar-brand" href="/">
                    <i class="fa-solid fa-face-smile me-2"></i>FaceAI
                </a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav me-auto">
                        <li class="nav-item">
                            <a class="nav-link {{ 'active' if active_page == 'home' }}" href="/">Home</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link {{ 'active' if active_page == 'camera' }}" href="/live_camera">Live Camera</a>
                        </li>
                        {% if is_admin %}
                        <li class="nav-item">
                            <a class="nav-link" href="/admin">Admin Panel</a>
                        </li>
                        {% endif %}
                    </ul>
                    <ul class="navbar-nav">
                        {% if 'user_id' in session %}
                        <li class="nav-item">
                            <span class="navbar-text me-3">Welcome, {{ session['username'] }}</span>
                        </li>
                        <li class="nav-item">
                            <a class="btn btn-outline-light" href="/logout">Logout</a>
                        </li>
                        {% else %}
                        <li class="nav-item">
                            <a class="btn btn-outline-light me-2" href="/login">Login</a>
                        </li>
                        <li class="nav-item">
                            <a class="btn btn-primary" href="/register">Register</a>
                        </li>
                        {% endif %}
                    </ul>
                </div>
            </div>
        </nav>
        {% if is_admin %}
        <div class="admin-link">
            <a href="/admin" class="btn btn-sm btn-outline-secondary">Admin Panel</a>
        </div>
        {% endif %}
        <div class="container-fluid mt-4"> <!-- Use container-fluid for full width -->
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                    <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            {% block content %}{% endblock %}
        </div>
        <footer class="bg-dark text-white mt-5 py-4">
            <div class="container text-center">
                <p>&copy; 2025 Face Recognition App. All rights reserved.</p> <!-- Updated Year -->
            </div>
        </footer>

        <!-- Custom Popup Structure -->
        <div class="custom-popup-overlay" id="customPopupOverlay">
            <div class="custom-popup">
                <div class="custom-popup-header">
                    <h5 class="custom-popup-title" id="customPopupTitle">Notification</h5>
                    <button class="custom-popup-close" id="customPopupClose">&times;</button>
                </div>
                <div class="custom-popup-body">
                    <p class="custom-popup-message" id="customPopupMessage"></p>
                </div>
                <div class="custom-popup-footer">
                    <button class="custom-popup-btn custom-popup-btn-primary" id="customPopupOk">OK</button>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // --- Custom Popup Functions ---
            const popupOverlay = document.getElementById('customPopupOverlay');
            const popupTitle = document.getElementById('customPopupTitle');
            const popupMessage = document.getElementById('customPopupMessage');
            const popupCloseBtn = document.getElementById('customPopupClose');
            const popupOkBtn = document.getElementById('customPopupOk');

            function showCustomPopup(message, type = 'info', title = 'Notification') {
                popupTitle.textContent = title;
                popupMessage.textContent = message;
                popupMessage.className = 'custom-popup-message'; // Reset classes
                if (type) {
                    popupMessage.classList.add(type); // Add type class for color
                }
                popupOverlay.style.display = 'flex';

                // Focus the OK button for keyboard accessibility
                popupOkBtn.focus();
            }

            function hideCustomPopup() {
                popupOverlay.style.display = 'none';
            }

            // Close popup on clicking X, OK button, or overlay background
            popupCloseBtn.addEventListener('click', hideCustomPopup);
            popupOkBtn.addEventListener('click', hideCustomPopup);
            popupOverlay.addEventListener('click', function(event) {
                if (event.target === popupOverlay) {
                    hideCustomPopup();
                }
            });

            // Allow closing with Escape key
            document.addEventListener('keydown', function(event) {
                if (event.key === 'Escape' && popupOverlay.style.display === 'flex') {
                    hideCustomPopup();
                }
            });
        </script>
        {% block scripts %}{% endblock %}
    </body>
    </html>
    ''',
    'index.html': '''
    {% extends "base.html" %}
    {% set active_page = 'home' %}
    {% block content %}
    <div class="text-center mb-4">
        <h1 class="display-4">Face Recognition Studio</h1>
        <p class="lead">Explore AI-powered face processing features</p>
    </div>
    <ul class="nav nav-tabs mb-4" id="featuresTab" role="tablist">
        <li class="nav-item" role="presentation">
            <button class="nav-link active" id="add-tab" data-bs-toggle="tab" data-bs-target="#add" type="button">Add Face</button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="recognize-tab" data-bs-toggle="tab" data-bs-target="#recognize" type="button">Recognize</button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="filter-tab" data-bs-toggle="tab" data-bs-target="#filter" type="button">Filters</button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="swap-tab" data-bs-toggle="tab" data-bs-target="#swap" type="button">Face Swap</button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="bgremove-tab" data-bs-toggle="tab" data-bs-target="#bgremove" type="button">BG Remove</button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="roast-tab" data-bs-toggle="tab" data-bs-target="#roast" type="button">Roast</button>
        </li>
    </ul>
    <div class="tab-content" id="featuresTabContent">
        <!-- Add Face Tab -->
        <div class="tab-pane fade show active" id="add" role="tabpanel">
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Register New Face</h5>
                            <form id="addFaceForm" enctype="multipart/form-data">
                                <div class="mb-3">
                                    <label for="faceName" class="form-label">Name</label>
                                    <input type="text" class="form-control" id="faceName" name="name" required>
                                </div>
                                <div class="mb-3">
                                    <label for="addFaceImage" class="form-label">Face Image</label>
                                    <input class="form-control" type="file" id="addFaceImage" name="image" accept="image/*">
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Or use Live Camera:</label>
                                    <div class="d-grid gap-2">
                                        <button type="button" id="captureAddFaceBtn" class="btn btn-outline-secondary capture-trigger" data-tool="add_face">
                                            <i class="fas fa-camera me-1"></i> Capture from Camera
                                        </button>
                                    </div>
                                </div>
                                <div class="d-grid">
                                    <button type="submit" class="btn btn-primary">Add Face</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Preview</h5>
                            <div class="text-center">
                                <img id="addPreview" src="#" class="img-fluid rounded d-none" alt="Preview">
                                <div id="addPlaceholder" class="border rounded p-5 text-center">
                                    <i class="fas fa-user fa-4x text-muted mb-3"></i>
                                    <p class="text-muted">Image preview will appear here</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- Recognize Tab -->
        <div class="tab-pane fade" id="recognize" role="tabpanel">
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Recognize Faces</h5>
                            <form id="recognizeForm" enctype="multipart/form-data">
                                <div class="mb-3">
                                    <label for="recognizeImage" class="form-label">Upload Image</label>
                                    <input class="form-control" type="file" id="recognizeImage" name="image" accept="image/*">
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Or use Live Camera:</label>
                                    <div class="d-grid gap-2">
                                        <button type="button" id="captureRecognizeBtn" class="btn btn-outline-secondary capture-trigger" data-tool="recognize_face">
                                            <i class="fas fa-camera me-1"></i> Capture from Camera
                                        </button>
                                    </div>
                                </div>
                                <div class="d-grid">
                                    <button type="submit" class="btn btn-primary">Recognize Faces</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Results</h5>
                            <div id="recognizeResults" class="text-center">
                                <div class="border rounded p-5 text-center">
                                    <i class="fas fa-search fa-4x text-muted mb-3"></i>
                                    <p class="text-muted">Recognition results will appear here</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- Filters Tab -->
        <div class="tab-pane fade" id="filter" role="tabpanel">
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Apply Filter</h5>
                            <form id="filterForm" enctype="multipart/form-data">
                                <div class="mb-3">
                                    <label for="filterImage" class="form-label">Upload Image</label>
                                    <input class="form-control" type="file" id="filterImage" name="image" accept="image/*">
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Select Filter</label>
                                    <div class="d-flex flex-wrap gap-2">
                                        <input type="radio" class="btn-check" name="filter_type" id="filterGlasses" value="glasses" autocomplete="off" checked>
                                        <label class="btn btn-outline-primary" for="filterGlasses">
                                            <i class="fas fa-glasses me-1"></i> Glasses
                                        </label>
                                        <input type="radio" class="btn-check" name="filter_type" id="filterMustache" value="mustache" autocomplete="off">
                                        <label class="btn btn-outline-primary" for="filterMustache">
                                            <i class="fas fa-mustache me-1"></i> Mustache
                                        </label>
                                        <input type="radio" class="btn-check" name="filter_type" id="filterHat" value="hat" autocomplete="off">
                                        <label class="btn btn-outline-primary" for="filterHat">
                                            <i class="fas fa-hat-cowboy me-1"></i> Hat
                                        </label>
                                        <input type="radio" class="btn-check" name="filter_type" id="filterDog" value="dog_nose" autocomplete="off">
                                        <label class="btn btn-outline-primary" for="filterDog">
                                            <i class="fas fa-paw me-1"></i> Dog Nose
                                        </label>
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Or use Live Camera:</label>
                                    <div class="d-grid gap-2">
                                        <button type="button" id="captureFilterBtn" class="btn btn-outline-secondary capture-trigger" data-tool="apply_filter">
                                            <i class="fas fa-camera me-1"></i> Capture from Camera
                                        </button>
                                    </div>
                                </div>
                                <div class="d-grid">
                                    <button type="submit" class="btn btn-primary">Apply Filter</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Filtered Image</h5>
                            <div class="text-center">
                                <img id="filterPreview" src="#" class="img-fluid rounded d-none" alt="Filtered Preview">
                                <div id="filterPlaceholder" class="border rounded p-5 text-center">
                                    <i class="fas fa-magic fa-4x text-muted mb-3"></i>
                                    <p class="text-muted">Filtered image will appear here</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- Face Swap Tab -->
        <div class="tab-pane fade" id="swap" role="tabpanel">
             <div class="alert alert-info">
                 <strong>Note:</strong> Face Swap works best with two clear, front-facing photos. Live Camera capture will take two consecutive frames.
             </div>
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Face Swap</h5>
                            <form id="swapForm" enctype="multipart/form-data">
                                <div class="row mb-3">
                                    <div class="col-md-6">
                                        <label for="swapImage1" class="form-label">First Face</label>
                                        <input class="form-control" type="file" id="swapImage1" name="image1" accept="image/*">
                                    </div>
                                    <div class="col-md-6">
                                        <label for="swapImage2" class="form-label">Second Face</label>
                                        <input class="form-control" type="file" id="swapImage2" name="image2" accept="image/*">
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Or use Live Camera:</label>
                                    <div class="d-grid gap-2">
                                        <button type="button" id="captureSwapBtn" class="btn btn-outline-secondary" onclick="startFaceSwapCapture()">
                                            <i class="fas fa-camera me-1"></i> Capture Two Frames
                                        </button>
                                    </div>
                                </div>
                                <div class="d-grid">
                                    <button type="submit" class="btn btn-primary">Swap Faces</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Result</h5>
                            <div class="text-center">
                                <img id="swapPreview" src="#" class="img-fluid rounded d-none" alt="Swapped Preview">
                                <div id="swapPlaceholder" class="border rounded p-5 text-center">
                                    <i class="fas fa-exchange-alt fa-4x text-muted mb-3"></i>
                                    <p class="text-muted">Face swap result will appear here</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- BG Remove Tab -->
        <div class="tab-pane fade" id="bgremove" role="tabpanel">
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Remove Background</h5>
                            <form id="bgremoveForm" enctype="multipart/form-data">
                                <div class="mb-3">
                                    <label for="bgremoveImage" class="form-label">Upload Image</label>
                                    <input class="form-control" type="file" id="bgremoveImage" name="image" accept="image/*">
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Or use Live Camera:</label>
                                    <div class="d-grid gap-2">
                                        <button type="button" id="captureBgRemoveBtn" class="btn btn-outline-secondary capture-trigger" data-tool="remove_background">
                                            <i class="fas fa-camera me-1"></i> Capture from Camera
                                        </button>
                                    </div>
                                </div>
                                <div class="d-grid">
                                    <button type="submit" class="btn btn-primary">Remove Background</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Result</h5>
                            <div class="text-center">
                                <img id="bgremovePreview" src="#" class="img-fluid rounded d-none" alt="BG Removed Preview">
                                <div id="bgremovePlaceholder" class="border rounded p-5 text-center">
                                    <i class="fas fa-layer-group fa-4x text-muted mb-3"></i>
                                    <p class="text-muted">Background removed image will appear here</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- Roast Tab -->
        <div class="tab-pane fade" id="roast" role="tabpanel">
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Roast My Image</h5>
                             <form id="roastForm" enctype="multipart/form-data">
                                <div class="mb-3">
                                    <label for="roastImage" class="form-label">Upload Image</label>
                                    <input class="form-control" type="file" id="roastImage" name="image" accept="image/*">
                                </div>
                                 <div class="mb-3">
                                    <label class="form-label">Or use Live Camera:</label>
                                     <div class="d-grid gap-2">
                                        <button type="button" id="captureRoastBtn" class="btn btn-outline-secondary capture-trigger" data-tool="roast_image">
                                            <i class="fas fa-camera me-1"></i> Capture from Camera
                                        </button>
                                    </div>
                                </div>
                                <div class="d-grid">
                                    <button type="submit" class="btn btn-primary">Get Roasted</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card h-100"> <!-- Added h-100 for consistent height -->
                        <div class="card-body">
                            <h5 class="card-title">Roast Result</h5>
                            <div id="roastResults" class="text-center p-4">
                                <!-- Removed placeholder-icon and placeholder-text classes -->
                                <div class="border rounded p-5 text-center">
                                    <i class="fas fa-fire fa-4x text-muted mb-3"></i>
                                    <p class="text-muted">Your roast will appear here</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    {% endblock %}
    {% block scripts %}
    <script>
        // Add Face Preview
        document.getElementById('addFaceImage').addEventListener('change', function(e) {
            const preview = document.getElementById('addPreview');
            const placeholder = document.getElementById('addPlaceholder');
            if (this.files && this.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.classList.remove('d-none');
                    placeholder.classList.add('d-none');
                }
                reader.readAsDataURL(this.files[0]);
            }
        });

        // Generic form submission handler
        async function setupForm(formId, endpoint, previewId, placeholderId, resultContainerId) {
            const form = document.getElementById(formId);
            if (!form) return; // Guard clause if form doesn't exist on page

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(form);
                const submitBtn = form.querySelector('button[type="submit"]');
                const originalBtnText = submitBtn.innerHTML;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span> Processing...';
                submitBtn.disabled = true;

                // Special handling for forms that might use live camera capture data
                // Check if a specific capture button for this tool has stored data
                const toolNameMap = {
                    'addFaceForm': 'add_face',
                    'recognizeForm': 'recognize_face',
                    'filterForm': 'apply_filter',
                    'bgremoveForm': 'remove_background',
                    'roastForm': 'roast_image'
                };
                const toolName = toolNameMap[formId];
                if (toolName) {
                    const captureBtn = document.querySelector(`.capture-trigger[data-tool="${toolName}"]`);
                    if (captureBtn && captureBtn.dataset.capturedImage) {
                        // If no file uploaded but image was captured, send the base64 data
                        // Check if the corresponding file input is empty or doesn't exist
                        let fileInputName = 'image';
                        if (formId === 'addFaceForm') fileInputName = 'image';
                        else if (formId === 'recognizeForm') fileInputName = 'image';
                        else if (formId === 'filterForm') fileInputName = 'image';
                        else if (formId === 'bgremoveForm') fileInputName = 'image';
                        else if (formId === 'roastForm') fileInputName = 'image';

                        const fileInput = form.querySelector(`input[type="file"][name="${fileInputName}"]`);
                        // Prioritize captured image if file input is empty or not present
                        if (!fileInput || fileInput.files.length === 0) {
                            formData.append('image_data', captureBtn.dataset.capturedImage);
                        }
                        // Clear captured data after use (regardless of whether it was used or not, to avoid stale data)
                        delete captureBtn.dataset.capturedImage;
                        captureBtn.classList.remove('btn-success');
                        captureBtn.classList.add('btn-outline-secondary');
                        captureBtn.innerHTML = '<i class="fas fa-camera me-1"></i> Capture from Camera';
                    }
                }

                // Special handling for Swap form to include camera capture
                if (formId === 'swapForm') {
                     const fileInput1 = document.getElementById('swapImage1');
                     const fileInput2 = document.getElementById('swapImage2');
                     const captureBtn = document.getElementById('captureSwapBtn');
                     if (captureBtn && captureBtn.dataset.capturedImage1 && captureBtn.dataset.capturedImage2) {
                         // If no files uploaded but images were captured, send the base64 data
                         // Check if file inputs are empty or not present
                         if ((fileInput1 && fileInput1.files.length === 0) && (fileInput2 && fileInput2.files.length === 0)) {
                             formData.append('image_data1', captureBtn.dataset.capturedImage1);
                             formData.append('image_data2', captureBtn.dataset.capturedImage2);
                         }
                         // Clear captured data after use
                         delete captureBtn.dataset.capturedImage1;
                         delete captureBtn.dataset.capturedImage2;
                         captureBtn.classList.remove('btn-success');
                         captureBtn.classList.add('btn-outline-secondary');
                         captureBtn.innerHTML = '<i class="fas fa-camera me-1"></i> Capture Two Frames';
                     } else if (captureBtn && (!captureBtn.dataset.capturedImage1 || !captureBtn.dataset.capturedImage2)) {
                         // Handled by form validation on backend or user interaction
                     }
                     // If files were uploaded, they will be sent normally via FormData
                }


                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    if (response.ok) {
                         // Handle image preview updates (for add, filter, swap, bgremove)
                        if (previewId && (result.annotated_image || result.filtered_image || result.swapped_image || result.processed_image)) {
                            const preview = document.getElementById(previewId);
                            const placeholder = document.getElementById(placeholderId);
                            const imageUrl = result.annotated_image || result.filtered_image || result.swapped_image || result.processed_image;
                            if (imageUrl) {
                                preview.src = imageUrl;
                                preview.classList.remove('d-none');
                                placeholder.classList.add('d-none');
                            }
                        }

                        // Handle specific result container updates
                        if (resultContainerId === 'recognizeResults' && result.names) {
                            const container = document.getElementById(resultContainerId);
                            let html = ``; // No image shown here in original, but could be added
                            if (result.names.length > 0) {
                                html += `<div class="alert alert-info">`;
                                result.names.forEach(name => {
                                    html += `<p class="mb-1"><strong>${name}</strong>`;
                                    if (result.emotion && result.emotion !== "Unknown") {
                                        html += ` - Feeling ${result.emotion}`;
                                    }
                                    html += `</p>`;
                                });
                                html += `</div>`;
                                 if(result.annotated_image) {
                                     html = `<img src="${result.annotated_image}" class="result-image mb-3">` + html;
                                 }
                            } else {
                                html += `<div class="alert alert-warning">No faces recognized</div>`;
                                if(result.annotated_image) {
                                     html = `<img src="${result.annotated_image}" class="result-image mb-3">` + html;
                                 }
                            }
                            container.innerHTML = html;
                        }
                        else if (resultContainerId === 'roastResults' && result.roast) {
                            const container = document.getElementById(resultContainerId);
                            container.innerHTML = `
                                <div class="alert alert-warning">
                                    <h5>Your Roast:</h5>
                                    <p>${result.roast}</p>
                                </div>
                            `;
                        }
                        // For Add Face, show success message
                         else if (formId === 'addFaceForm' && result.message) {
                              const container = document.getElementById('addPlaceholder'); // Reuse placeholder area
                              container.innerHTML = `<div class="alert alert-success">${result.message}</div>`;
                              const preview = document.getElementById('addPreview');
                              preview.classList.add('d-none');
                         }


                    } else {
                        // Use custom popup for errors instead of alert
                        showCustomPopup(`Error: ${result.error || 'Something went wrong'}`, 'error', 'Error');
                         // Reset capture buttons on error if data was potentially used or just to be safe
                         Object.keys(toolNameMap).forEach(fId => {
                              if (fId === formId) {
                                  const tName = toolNameMap[fId];
                                   const cBtn = document.querySelector(`.capture-trigger[data-tool="${tName}"]`);
                                   if (cBtn) {
                                       delete cBtn.dataset.capturedImage;
                                       cBtn.classList.remove('btn-success');
                                       cBtn.classList.add('btn-outline-secondary');
                                       cBtn.innerHTML = '<i class="fas fa-camera me-1"></i> Capture from Camera';
                                   }
                              }
                         });
                         if (formId === 'swapForm') {
                              const cBtn = document.getElementById('captureSwapBtn');
                              if (cBtn) {
                                  delete cBtn.dataset.capturedImage1;
                                  delete cBtn.dataset.capturedImage2;
                                  cBtn.classList.remove('btn-success');
                                  cBtn.classList.add('btn-outline-secondary');
                                  cBtn.innerHTML = '<i class="fas fa-camera me-1"></i> Capture Two Frames';
                              }
                         }
                         // Clear file inputs on error
                         form.reset();
                         // Reset previews/placeholders
                         if (previewId) {
                             const preview = document.getElementById(previewId);
                             const placeholder = document.getElementById(placeholderId);
                             preview.classList.add('d-none');
                             placeholder.classList.remove('d-none');
                             if (previewId === 'addPreview') {
                                 placeholder.innerHTML = '<i class="fas fa-user fa-4x text-muted mb-3"></i><p class="text-muted">Image preview will appear here</p>';
                             } else if (previewId === 'filterPreview') {
                                 placeholder.innerHTML = '<i class="fas fa-magic fa-4x text-muted mb-3"></i><p class="text-muted">Filtered image will appear here</p>';
                             } else if (previewId === 'swapPreview') {
                                 placeholder.innerHTML = '<i class="fas fa-exchange-alt fa-4x text-muted mb-3"></i><p class="text-muted">Face swap result will appear here</p>';
                             } else if (previewId === 'bgremovePreview') {
                                 placeholder.innerHTML = '<i class="fas fa-layer-group fa-4x text-muted mb-3"></i><p class="text-muted">Background removed image will appear here</p>';
                             }
                         }
                         if (resultContainerId === 'recognizeResults') {
                             document.getElementById(resultContainerId).innerHTML = '<div class="border rounded p-5 text-center"><i class="fas fa-search fa-4x text-muted mb-3"></i><p class="text-muted">Recognition results will appear here</p></div>';
                         }
                         if (resultContainerId === 'roastResults') {
                             document.getElementById(resultContainerId).innerHTML = '<div class="border rounded p-5 text-center"><i class="fas fa-fire fa-4x text-muted mb-3"></i><p class="text-muted">Your roast will appear here</p></div>';
                         }
                    }
                } catch (error) {
                    console.error('Error:', error);
                    // Use custom popup for network errors
                    showCustomPopup('An error occurred. Please try again.', 'error', 'Network Error');
                     // Reset capture buttons on network error
                     Object.keys(toolNameMap).forEach(fId => {
                          if (fId === formId) {
                              const tName = toolNameMap[fId];
                               const cBtn = document.querySelector(`.capture-trigger[data-tool="${tName}"]`);
                               if (cBtn) {
                                   delete cBtn.dataset.capturedImage;
                                   cBtn.classList.remove('btn-success');
                                   cBtn.classList.add('btn-outline-secondary');
                                   cBtn.innerHTML = '<i class="fas fa-camera me-1"></i> Capture from Camera';
                               }
                          }
                     });
                     if (formId === 'swapForm') {
                          const cBtn = document.getElementById('captureSwapBtn');
                          if (cBtn) {
                              delete cBtn.dataset.capturedImage1;
                              delete cBtn.dataset.capturedImage2;
                              cBtn.classList.remove('btn-success');
                              cBtn.classList.add('btn-outline-secondary');
                              cBtn.innerHTML = '<i class="fas fa-camera me-1"></i> Capture Two Frames';
                          }
                     }
                     form.reset();
                     // Reset previews/placeholders
                     if (previewId) {
                         const preview = document.getElementById(previewId);
                         const placeholder = document.getElementById(placeholderId);
                         preview.classList.add('d-none');
                         placeholder.classList.remove('d-none');
                         if (previewId === 'addPreview') {
                             placeholder.innerHTML = '<i class="fas fa-user fa-4x text-muted mb-3"></i><p class="text-muted">Image preview will appear here</p>';
                         } else if (previewId === 'filterPreview') {
                             placeholder.innerHTML = '<i class="fas fa-magic fa-4x text-muted mb-3"></i><p class="text-muted">Filtered image will appear here</p>';
                         } else if (previewId === 'swapPreview') {
                             placeholder.innerHTML = '<i class="fas fa-exchange-alt fa-4x text-muted mb-3"></i><p class="text-muted">Face swap result will appear here</p>';
                         } else if (previewId === 'bgremovePreview') {
                             placeholder.innerHTML = '<i class="fas fa-layer-group fa-4x text-muted mb-3"></i><p class="text-muted">Background removed image will appear here</p>';
                         }
                     }
                     if (resultContainerId === 'recognizeResults') {
                         document.getElementById(resultContainerId).innerHTML = '<div class="border rounded p-5 text-center"><i class="fas fa-search fa-4x text-muted mb-3"></i><p class="text-muted">Recognition results will appear here</p></div>';
                     }
                     if (resultContainerId === 'roastResults') {
                         document.getElementById(resultContainerId).innerHTML = '<div class="border rounded p-5 text-center"><i class="fas fa-fire fa-4x text-muted mb-3"></i><p class="text-muted">Your roast will appear here</p></div>';
                     }
                } finally {
                    submitBtn.innerHTML = originalBtnText;
                    submitBtn.disabled = false;
                }
            });
        }

        // --- Live Camera Capture Logic for Individual Tools ---
        document.addEventListener('DOMContentLoaded', function() {
            const captureButtons = document.querySelectorAll('.capture-trigger');
            captureButtons.forEach(button => {
                button.addEventListener('click', async function() {
                    const tool = this.dataset.tool;
                    if (!tool) return;

                    const originalBtnText = this.innerHTML;
                    this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span> Capturing...';
                    this.disabled = true;

                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                        const video = document.createElement('video');
                        video.srcObject = stream;
                        await video.play();

                        // Wait a bit for the camera to adjust
                        await new Promise(resolve => setTimeout(resolve, 500));

                        const canvas = document.createElement('canvas');
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const imageDataUrl = canvas.toDataURL('image/jpeg');

                        stream.getTracks().forEach(track => track.stop()); // Stop the camera stream

                        // Store the captured image data in the button's dataset
                        this.dataset.capturedImage = imageDataUrl;
                        this.classList.remove('btn-outline-secondary');
                        this.classList.add('btn-success'); // Visual feedback
                        this.innerHTML = '<i class="fas fa-check me-1"></i> Captured!';

                        // Optional: Show preview somewhere?
                        // const previewArea = document.getElementById('some-preview-div');
                        // if (previewArea) {
                        //     previewArea.innerHTML = `<img src="${imageDataUrl}" class="img-fluid">`;
                        // }

                    } catch (err) {
                        console.error("Error accessing camera for capture:", err);
                        // Use custom popup for camera errors
                        showCustomPopup("Could not access the camera. Please check permissions.", 'warning', 'Camera Access Denied');
                        this.innerHTML = originalBtnText;
                    } finally {
                        this.disabled = false;
                    }
                });
            });
        });

        // --- Face Swap Specific Capture Logic ---
        let faceSwapCaptureCount = 0;
        let faceSwapImageData1 = null;
        let faceSwapImageData2 = null;
        function startFaceSwapCapture() {
            faceSwapCaptureCount = 0;
            faceSwapImageData1 = null;
            faceSwapImageData2 = null;
            captureFaceSwapFrame();
        }

        async function captureFaceSwapFrame() {
            const captureBtn = document.getElementById('captureSwapBtn');
            if (!captureBtn) return;

            const originalBtnText = captureBtn.innerHTML;
            captureBtn.innerHTML = `<span class="spinner-border spinner-border-sm" role="status"></span> Capturing Frame ${faceSwapCaptureCount + 1}...`;
            captureBtn.disabled = true;

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                const video = document.createElement('video');
                video.srcObject = stream;
                await video.play();

                // Wait a bit for the camera to adjust
                await new Promise(resolve => setTimeout(resolve, 500));

                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageDataUrl = canvas.toDataURL('image/jpeg');

                stream.getTracks().forEach(track => track.stop()); // Stop the camera stream

                faceSwapCaptureCount++;
                if (faceSwapCaptureCount === 1) {
                    faceSwapImageData1 = imageDataUrl;
                    captureBtn.innerHTML = '<i class="fas fa-camera me-1"></i> Capture Second Frame';
                    captureBtn.disabled = false;
                    // Brief delay before next capture prompt or automatic capture
                    setTimeout(() => {
                        if (faceSwapCaptureCount === 1) { // Only proceed if user hasn't clicked again
                           // Option 1: Ask user
                           if (confirm("Frame 1 captured. Click OK to capture Frame 2 now.")) {
                               captureFaceSwapFrame();
                           } else {
                               // User cancelled, reset
                                faceSwapCaptureCount = 0;
                                faceSwapImageData1 = null;
                                faceSwapImageData2 = null;
                                captureBtn.innerHTML = '<i class="fas fa-camera me-1"></i> Capture Two Frames';
                                captureBtn.classList.remove('btn-success');
                                captureBtn.classList.add('btn-outline-secondary');
                           }
                           // Option 2: Auto-capture (uncomment below, comment above if block)
                           // captureFaceSwapFrame();
                        }
                    }, 1000); // 1 second delay
                } else if (faceSwapCaptureCount === 2) {
                    faceSwapImageData2 = imageDataUrl;
                    // Store data in button dataset for form submission
                    captureBtn.dataset.capturedImage1 = faceSwapImageData1;
                    captureBtn.dataset.capturedImage2 = faceSwapImageData2;
                    captureBtn.classList.remove('btn-outline-secondary');
                    captureBtn.classList.add('btn-success'); // Visual feedback
                    captureBtn.innerHTML = '<i class="fas fa-check-double me-1"></i> Two Frames Captured!';
                }

            } catch (err) {
                console.error("Error accessing camera for Face Swap capture:", err);
                // Use custom popup for camera errors
                showCustomPopup("Could not access the camera for Face Swap. Please check permissions.", 'warning', 'Camera Access Denied');
                faceSwapCaptureCount = 0; // Reset on error
                faceSwapImageData1 = null;
                faceSwapImageData2 = null;
                captureBtn.innerHTML = originalBtnText;
                captureBtn.disabled = false;
                // Clear dataset on error
                delete captureBtn.dataset.capturedImage1;
                delete captureBtn.dataset.capturedImage2;
            }
        }

        // Set up all forms (ensure these IDs exist on the respective pages)
        document.addEventListener('DOMContentLoaded', function() {
             setupForm('addFaceForm', '/add_face', 'addPreview', 'addPlaceholder');
             setupForm('recognizeForm', '/recognize_face', null, null, 'recognizeResults'); // Preview handled in result
             setupForm('filterForm', '/apply_filter', 'filterPreview', 'filterPlaceholder');
             setupForm('swapForm', '/face_swap', 'swapPreview', 'swapPlaceholder');
             setupForm('bgremoveForm', '/remove_background', 'bgremovePreview', 'bgremovePlaceholder');
             setupForm('roastForm', '/roast_image', null, null, 'roastResults');
        });

    </script>
    {% endblock %}
    ''',
    'live_camera.html': '''
    {% extends "base.html" %}
    {% set active_page = 'camera' %}
    {% block content %}
    <div class="text-center mb-3">
        <h2>Live Face Recognition</h2>
        <p class="text-muted">Real-time face detection and recognition</p>
    </div>
    <div class="camera-container mb-4">
        <img id="videoFeed" src="{{ url_for('video_feed') }}">
        <div class="scanner-line" id="scannerLine"></div>
        <div class="processing-overlay" id="processingOverlay">
            <div class="text-center">
                <div class="spinner-border text-light" role="status">
                    <span class="visually-hidden">Processing...</span>
                </div>
                <p class="mt-2 mb-0">Processing frame...</p>
            </div>
        </div>
        <!-- Tool selection buttons for live camera -->
        <div id="liveCameraTools">
             <button class="btn btn-sm btn-primary tool-btn" data-tool="recognize_face">Recognize</button>
             <button class="btn btn-sm btn-success tool-btn" data-tool="apply_filter" data-filter="glasses">Glasses</button>
             <button class="btn btn-sm btn-warning tool-btn" data-tool="apply_filter" data-filter="mustache">Stache</button>
             <button class="btn btn-sm btn-info tool-btn" data-tool="remove_background">BG Remove</button>
             <button class="btn btn-sm btn-danger tool-btn" data-tool="roast_image">Roast</button>
        </div>
        <button class="btn btn-primary capture-btn" id="liveCaptureBtn">
             <i class="fas fa-camera"></i> Capture & Process
        </button>
    </div>
    <div class="container-fluid"> <!-- Use container-fluid for full width results -->
         <div class="row">
             <div class="col-12">
                 <div class="card">
                     <div class="card-body">
                         <h5 class="card-title">Live Tool Result</h5>
                         <div id="liveCameraResult" class="text-center">
                             <!-- Removed placeholder-icon and placeholder-text classes -->
                             <div class="border rounded p-5 text-center">
                                 <i class="fas fa-cogs fa-4x text-muted mb-3"></i>
                                 <p class="text-muted">Processed results will appear here</p>
                             </div>
                         </div>
                     </div>
                 </div>
             </div>
         </div>
    </div>
    {% endblock %}
    {% block scripts %}
    <script>
        let captureInterval = null;
        let isCapturing = false;
        const videoFeed = document.getElementById('videoFeed');
        const scannerLine = document.getElementById('scannerLine');
        const processingOverlay = document.getElementById('processingOverlay');
        const liveCaptureBtn = document.getElementById('liveCaptureBtn');
        const liveCameraTools = document.getElementById('liveCameraTools');
        const liveCameraResult = document.getElementById('liveCameraResult');
        let selectedTool = null;
        let selectedFilter = null; // For filter tool

        // Function to capture frame and send for processing
        async function captureAndProcess() {
            if (isCapturing) return; // Prevent overlapping captures
            if (!selectedTool) {
                showCustomPopup("Please select a tool first.", 'warning', 'No Tool Selected');
                return;
            }
            isCapturing = true;
            scannerLine.style.display = 'block';
            processingOverlay.style.display = 'flex';

            try {
                const canvas = document.createElement('canvas');
                canvas.width = videoFeed.videoWidth || videoFeed.naturalWidth;
                canvas.height = videoFeed.videoHeight || videoFeed.naturalHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
                const imageDataUrl = canvas.toDataURL('image/jpeg');

                const formData = new FormData();
                formData.append('image_data', imageDataUrl);

                // Add tool-specific data
                let endpoint = '';
                if (selectedTool === 'recognize_face') {
                    endpoint = '/recognize_face';
                } else if (selectedTool === 'apply_filter') {
                    endpoint = '/apply_filter';
                    formData.append('filter_type', selectedFilter || 'glasses'); // Default to glasses
                } else if (selectedTool === 'remove_background') {
                    endpoint = '/remove_background';
                } else if (selectedTool === 'roast_image') {
                    endpoint = '/roast_image';
                } else {
                    throw new Error('Unknown tool selected');
                }

                const response = await fetch(endpoint, {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                     if (result.annotated_image || result.filtered_image || result.processed_image) {
                         const imageUrl = result.annotated_image || result.filtered_image || result.processed_image;
                         liveCameraResult.innerHTML = `<img src="${imageUrl}" class="img-fluid rounded" alt="Processed Image">`;
                     } else if (result.roast) {
                         liveCameraResult.innerHTML = `
                             <div class="alert alert-warning">
                                 <h5>Your Roast:</h5>
                                 <p>${result.roast}</p>
                             </div>
                         `;
                     } else if (result.names) {
                         // Handle recognize result without image if needed, but usually annotated_image is present
                          let html = ``;
                          if (result.names.length > 0) {
                              html += `<div class="alert alert-info">`;
                              result.names.forEach(name => {
                                  html += `<p class="mb-1"><strong>${name}</strong>`;
                                  if (result.emotion && result.emotion !== "Unknown") {
                                      html += ` - Feeling ${result.emotion}`;
                                  }
                                  html += `</p>`;
                              });
                              html += `</div>`;
                               if(result.annotated_image) {
                                   html = `<img src="${result.annotated_image}" class="img-fluid rounded mb-3">` + html;
                               }
                          } else {
                              html += `<div class="alert alert-warning">No faces recognized</div>`;
                              if(result.annotated_image) {
                                   html = `<img src="${result.annotated_image}" class="img-fluid rounded mb-3">` + html;
                               }
                          }
                          liveCameraResult.innerHTML = html;
                     } else {
                         liveCameraResult.innerHTML = `<div class="alert alert-info">Processing complete, but no image returned.</div>`;
                     }
                } else {
                    throw new Error(result.error || 'Processing failed');
                }
            } catch (error) {
                console.error('Live capture error:', error);
                // Use custom popup for errors
                showCustomPopup(`Error: ${error.message}`, 'error', 'Processing Error');
                liveCameraResult.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            } finally {
                isCapturing = false;
                scannerLine.style.display = 'none';
                processingOverlay.style.display = 'none';
            }
        }

        // Tool selection
        document.querySelectorAll('.tool-btn').forEach(button => {
            button.addEventListener('click', function() {
                document.querySelectorAll('.tool-btn').forEach(btn => btn.classList.remove('active'));
                this.classList.add('active');
                selectedTool = this.dataset.tool;
                selectedFilter = this.dataset.filter || null; // Get filter type if applicable
                liveCaptureBtn.style.display = 'block';
                liveCameraTools.style.display = 'block'; // Keep tools visible
                 // Reset result area when tool changes
                 liveCameraResult.innerHTML = '<div class="border rounded p-5 text-center"><i class="fas fa-cogs fa-4x text-muted mb-3"></i><p class="text-muted">Processed results will appear here</p></div>';
            });
        });

        // Manual capture button
        liveCaptureBtn.addEventListener('click', captureAndProcess);

        // Auto-capture every 2 seconds (example)
        // Uncomment the lines below to enable auto-capture
        /*
        function startAutoCapture() {
            if (captureInterval) clearInterval(captureInterval);
            captureInterval = setInterval(captureAndProcess, 2000); // Every 2 seconds
        }
        function stopAutoCapture() {
            if (captureInterval) {
                clearInterval(captureInterval);
                captureInterval = null;
            }
        }
        // Example: Start auto-capture when a tool is selected and stop when none is selected
        // You can add logic to start/stop based on user interaction or a toggle button
        */

        // Simple connection status indicator (unchanged)
        const statusEl = document.getElementById('recognitionStatus');
        if (statusEl) { // Only run if status element exists (might not be present in this specific template anymore)
            let connected = false;
            function checkConnection() {
                fetch('/video_feed')
                    .then(response => {
                        if (response.ok && !connected) {
                            connected = true;
                            statusEl.innerHTML = `
                                <div class="alert alert-success">
                                    <i class="fas fa-check-circle me-2"></i>
                                    Camera connected. Faces will be recognized in real-time
                                </div>
                                <p class="mt-2 text-muted">Look directly at the camera for best results</p>
                            `;
                        } else if (!response.ok && connected) {
                            connected = false;
                            statusEl.innerHTML = `
                                <div class="alert alert-danger">
                                    <i class="fas fa-exclamation-triangle me-2"></i>
                                    Camera disconnected. Trying to reconnect...
                                </div>
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                            `;
                        }
                    })
                    .catch(() => {
                        if (connected) {
                            connected = false;
                            statusEl.innerHTML = `
                                <div class="alert alert-danger">
                                    <i class="fas fa-exclamation-triangle me-2"></i>
                                    Camera disconnected. Trying to reconnect...
                                </div>
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                            `;
                        }
                    });
            }
            // Check connection every 5 seconds
            setInterval(checkConnection, 5000);
            checkConnection();
        }
    </script>
    {% endblock %}
    ''',
    'register.html': '''
    {% extends "base.html" %}
    {% block content %}
    <div class="row justify-content-center">
        <div class="col-md-6">
            <div class="card shadow">
                <div class="card-body">
                    <h2 class="card-title text-center mb-4">Create Account</h2>
                    <form method="POST" action="{{ url_for('register') }}">
                        <div class="mb-3">
                            <label for="username" class="form-label">Username</label>
                            <input type="text" class="form-control" id="username" name="username" required>
                        </div>
                        <div class="mb-3">
                            <label for="email" class="form-label">Email</label>
                            <input type="email" class="form-control" id="email" name="email" required>
                        </div>
                        <div class="mb-3">
                            <label for="password" class="form-label">Password</label>
                            <input type="password" class="form-control" id="password" name="password" required>
                        </div>
                        <div class="d-grid mb-3">
                            <button type="submit" class="btn btn-primary btn-lg">Register</button>
                        </div>
                        <div class="text-center">
                            <p class="mb-0">Already have an account? <a href="{{ url_for('login') }}">Login</a></p>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>
    {% endblock %}
    ''',
    'login.html': '''
    {% extends "base.html" %}
    {% block content %}
    <div class="row justify-content-center">
        <div class="col-md-6">
            <div class="card shadow">
                <div class="card-body">
                    <h2 class="card-title text-center mb-4">Login</h2>
                    <form method="POST" action="{{ url_for('login') }}">
                        <div class="mb-3">
                            <label for="username" class="form-label">Username</label>
                            <input type="text" class="form-control" id="username" name="username" required>
                        </div>
                        <div class="mb-3">
                            <label for="password" class="form-label">Password</label>
                            <input type="password" class="form-control" id="password" name="password" required>
                        </div>
                        <div class="d-grid mb-3">
                            <button type="submit" class="btn btn-primary btn-lg">Login</button>
                        </div>
                        <div class="text-center">
                            <p class="mb-0">Don't have an account? <a href="{{ url_for('register') }}">Register</a></p>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>
    {% endblock %}
    ''',
    # --- Admin Template ---
    'admin.html': '''
    {% extends "base.html" %}
    {% block content %}
    <div class="container">
        <h2 class="my-4">Admin Panel</h2>

        {% if faces %}
        <div class="card mb-4">
            <div class="card-header">
                <h5>Manage Registered Faces</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped table-hover">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Name</th>
                                <th>Added By</th>
                                <th>Date Added</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for face in faces %}
                            <tr>
                                <td>{{ face[0] }}</td>
                                <td>{{ face[1] }}</td>
                                <td>{{ face[3] }}</td>
                                <td>{{ face[2] }}</td>
                                <td>
                                    <button class="btn btn-danger btn-sm delete-face-btn" data-face-id="{{ face[0] }}">Delete</button>
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        {% else %}
        <p>No faces registered yet.</p>
        {% endif %}

        <div class="card">
            <div class="card-header">
                <h5>Manage Banned IPs</h5>
            </div>
            <div class="card-body">
                 <form id="banIpForm" class="mb-3">
                    <div class="row g-2">
                        <div class="col-md-5">
                            <input type="text" class="form-control" id="ipAddress" name="ip_address" placeholder="Enter IP address" required>
                        </div>
                        <div class="col-md-5">
                            <input type="text" class="form-control" id="banReason" name="reason" placeholder="Reason (optional)">
                        </div>
                        <div class="col-md-2">
                            <button type="submit" class="btn btn-warning w-100">Ban IP</button>
                        </div>
                    </div>
                </form>
                {% if banned_ips %}
                <div class="table-responsive">
                    <table class="table table-striped table-hover">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>IP Address</th>
                                <th>Reason</th>
                                <th>Banned At</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for ban in banned_ips %}
                            <tr>
                                <td>{{ ban[0] }}</td>
                                <td>{{ ban[1] }}</td>
                                <td>{{ ban[2] or 'No reason provided' }}</td>
                                <td>{{ ban[3] }}</td>
                                <td>
                                    <button class="btn btn-success btn-sm unban-ip-btn" data-ban-id="{{ ban[0] }}">Unban</button>
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                {% else %}
                <p>No IPs are currently banned.</p>
                {% endif %}
            </div>
        </div>
    </div>
    {% endblock %}
    {% block scripts %}
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Delete Face
            document.querySelectorAll('.delete-face-btn').forEach(button => {
                button.addEventListener('click', async function() {
                    const faceId = this.dataset.faceId;
                    // Use custom popup for confirmation
                    showCustomPopup(`Are you sure you want to delete face ID ${faceId}?`, 'warning', 'Confirm Delete', true); // true for confirm mode

                    // Handle confirmation
                    const handleConfirm = () => {
                        hideCustomPopup(); // Hide the confirmation popup first
                        // Proceed with deletion
                        (async () => {
                            try {
                                const response = await fetch(`/admin/delete_face/${faceId}`, {
                                    method: 'POST',
                                    headers: {
                                        'X-Requested-With': 'XMLHttpRequest' // Indicate AJAX
                                    }
                                    // No body needed for POST in this simple case, but CSRF token would be good in production
                                });
                                const result = await response.json();
                                if (response.ok) {
                                    showCustomPopup(result.message, 'success', 'Success'); // Show success message
                                    // Reload the page or remove the row dynamically
                                    location.reload(); // Simple reload
                                    // Or dynamically remove row:
                                    // this.closest('tr').remove();
                                } else {
                                    showCustomPopup('Error: ' + (result.error || 'Failed to delete face.'), 'error', 'Error'); // Show error
                                }
                            } catch (error) {
                                console.error('Error deleting face:', error);
                                showCustomPopup('An error occurred while deleting the face.', 'error', 'Error'); // Show error
                            }
                        })();
                    };

                    // Temporarily override the OK button's click handler for confirmation
                    const originalOkHandler = popupOkBtn.onclick;
                    popupOkBtn.onclick = handleConfirm;
                    popupOkBtn.textContent = 'Delete'; // Change button text

                    // Restore original handler and text when popup is closed
                    const closePopupAndRestore = () => {
                        popupOkBtn.onclick = originalOkHandler;
                        popupOkBtn.textContent = 'OK';
                        hideCustomPopup();
                        popupCloseBtn.removeEventListener('click', closePopupAndRestore);
                        popupOverlay.removeEventListener('click', closePopupAndRestore);
                        document.removeEventListener('keydown', handleEscapeKey);
                    };

                    const handleEscapeKey = (event) => {
                        if (event.key === 'Escape') {
                            closePopupAndRestore();
                        }
                    };

                    popupCloseBtn.addEventListener('click', closePopupAndRestore);
                    popupOverlay.addEventListener('click', closePopupAndRestore);
                    document.addEventListener('keydown', handleEscapeKey);
                });
            });

            // Ban IP
            document.getElementById('banIpForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                try {
                    const response = await fetch('/admin/ban_ip', {
                        method: 'POST',
                        body: formData,
                        headers: {
                            'X-Requested-With': 'XMLHttpRequest'
                        }
                    });
                    const result = await response.json();
                    if (response.ok) {
                        showCustomPopup(result.message, 'success', 'Success');
                        location.reload(); // Reload to show updated list
                    } else {
                        showCustomPopup('Error: ' + (result.error || 'Failed to ban IP.'), 'error', 'Error');
                    }
                } catch (error) {
                    console.error('Error banning IP:', error);
                    showCustomPopup('An error occurred while banning the IP.', 'error', 'Error');
                }
            });

            // Unban IP
            document.querySelectorAll('.unban-ip-btn').forEach(button => {
                button.addEventListener('click', async function() {
                    const banId = this.dataset.banId;
                    // Use custom popup for confirmation
                     showCustomPopup(`Are you sure you want to unban IP ban ID ${banId}?`, 'warning', 'Confirm Unban', true); // true for confirm mode

                    // Handle confirmation
                    const handleConfirm = () => {
                        hideCustomPopup(); // Hide the confirmation popup first
                        // Proceed with unban
                        (async () => {
                            try {
                                const response = await fetch(`/admin/unban_ip/${banId}`, {
                                    method: 'POST',
                                    headers: {
                                        'X-Requested-With': 'XMLHttpRequest'
                                    }
                                    // No body needed
                                });
                                const result = await response.json();
                                if (response.ok) {
                                    showCustomPopup(result.message, 'success', 'Success');
                                    location.reload(); // Reload to show updated list
                                } else {
                                    showCustomPopup('Error: ' + (result.error || 'Failed to unban IP.'), 'error', 'Error');
                                }
                            } catch (error) {
                                console.error('Error unbanning IP:', error);
                                showCustomPopup('An error occurred while unbanning the IP.', 'error', 'Error');
                            }
                        })();
                    };

                    // Temporarily override the OK button's click handler for confirmation
                    const originalOkHandler = popupOkBtn.onclick;
                    popupOkBtn.onclick = handleConfirm;
                    popupOkBtn.textContent = 'Unban'; // Change button text

                    // Restore original handler and text when popup is closed
                    const closePopupAndRestore = () => {
                        popupOkBtn.onclick = originalOkHandler;
                        popupOkBtn.textContent = 'OK';
                        hideCustomPopup();
                        popupCloseBtn.removeEventListener('click', closePopupAndRestore);
                        popupOverlay.removeEventListener('click', closePopupAndRestore);
                        document.removeEventListener('keydown', handleEscapeKey);
                    };

                    const handleEscapeKey = (event) => {
                        if (event.key === 'Escape') {
                            closePopupAndRestore();
                        }
                    };

                    popupCloseBtn.addEventListener('click', closePopupAndRestore);
                    popupOverlay.addEventListener('click', closePopupAndRestore);
                    document.addEventListener('keydown', handleEscapeKey);
                });
            });
        });
    </script>
    {% endblock %}
    '''
}
# Create Jinja environment with DictLoader
app.jinja_loader = DictLoader(templates_dict)

if __name__ == '__main__':
    app.run(debug=True) # Set debug=False in production
