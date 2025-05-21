# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
import json
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from src.run.main import infer, infer_img
import os
from enum import Enum
import cv2
import onnxruntime as ort
import google.generativeai as genai
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///business_card_ocr.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = SQLAlchemy(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

genai.configure(api_key=os.getenv("GEMINI_KEY"))
available_models = genai.list_models()

# Kiểm tra xem mô hình cần tìm có trong danh sách không
# llm = pipeline("text-generation", model="google/gemma-2-2b-it")
llm = "mèo"
for model in available_models:
    if model.name == "models/gemma-3-1b-it":
        llm = None
print("Use GEMINI API") if llm==None else print("Using LLM")


model_dir = os.path.join("model", "checkpoint-448")
label_list = ["O", "B-Name", "I-Name", "B-Position", "I-Position", "B-Company", 
              "I-Company", "B-Address", "I-Address", "B-Phone", "I-Phone", 
              "B-Email", "I-Email", "B-Department", "I-Department"]
              
label_to_id = {label: i for i, label in enumerate(label_list)}
model = AutoModelForTokenClassification.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# model = ort.InferenceSession("model/checkpoint-200/ner_model.onnx")
tokenizer = AutoTokenizer.from_pretrained(model_dir)


class Language(str, Enum):
    auto = "auto"
    english = "english"
    korean = "korean"
    japanese = "japanese"
    vietnamese = "vietnamese"
    unknown = "unknown"

# Define models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    business_cards = db.relationship('BusinessCard', backref='user', lazy=True)


class BusinessCard(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    original_filename = db.Column(db.String(200), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, processed, failed
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed_at = db.Column(db.DateTime)
    extracted_data = db.Column(db.Text)  # JSON data
    edited_data = db.Column(db.Text)  # JSON data after user edits
    logs = db.Column(db.Text)  # Processing logs

@app.template_filter('fromjson')
def fromjson_filter(s):
    return json.loads(s)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def extract_info_from_card(image_path, lang="vietnamese"):
    """Extract information from business card image using OCR"""
    try:
        # Read the image
        pil_image = Image.open(image_path)

        image = np.array(pil_image)

        # Kiểm tra kích thước để đảm bảo không bị lỗi (GIF có thể có kênh alpha)
        if image.shape[-1] == 4:  # RGBA
           image = image[:, :, :3]
        
        language = Language(lang)
        if language.value == "unknown":
            return {"Language": "Unknown"}
        re = infer_img(image, model, tokenizer, label_list, language.value, device, llm=llm)
        print(re)
        return {
            'success': True,
            'data': re,
            'log': f"Successfully extracted information from {image_path}"
        }
    except Exception as e:
        return {
            'success': False,
            'data': {},
            'log': f"Error extracting information: {str(e)}"
        }

# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'danger')
            return render_template('register.html')
        
        # Create new user
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)
        
        # Make the first user an admin
        if User.query.count() == 2:
            new_user.is_admin = True
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Redirect admin to admin dashboard
    if current_user.is_admin:
        return redirect(url_for('admin_dashboard'))
    
    # Get user's business cards
    cards = BusinessCard.query.filter_by(user_id=current_user.id).order_by(BusinessCard.uploaded_at.desc()).all()
    
    return render_template('dashboard.html', cards=cards)

@app.route('/upload', methods=['POST'])
@login_required
def upload_card():
    if 'card_image' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('dashboard'))
    
    file = request.files['card_image']
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('dashboard'))
    
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        # Generate a unique filename
        unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Create a new business card record with pending status
        new_card = BusinessCard(
            user_id=current_user.id,
            filename=unique_filename,
            original_filename=original_filename,
            status='pending'
        )
        
        db.session.add(new_card)
        db.session.commit()
        
        # Redirect to edit page without processing - we'll process on demand
        return redirect(url_for('edit_card', card_id=new_card.id))
    
    flash('Invalid file type', 'danger')
    return redirect(url_for('dashboard'))

@app.route('/extract/<int:card_id>', methods=['POST'])
@login_required
def extract_card_info(card_id):
    card = BusinessCard.query.get_or_404(card_id)
    
    # Ensure the card belongs to the current user
    if card.user_id != current_user.id:
        flash('You do not have permission to process this card', 'danger')
        return jsonify({'success': False, 'message': 'Permission denied'})
    
    # Get the language selection (default to English if not specified)
    lang = request.form.get('language', 'english')
    
    # Get the file path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], card.filename)
    
    # Process the card with the selected language
    result = extract_info_from_card(file_path, lang=lang)
    
    # Update the card record with the extracted information
    card.status = 'processed' if result['success'] else 'failed'
    card.processed_at = datetime.utcnow()
    card.extracted_data = json.dumps(result['data'])
    card.edited_data = json.dumps(result['data'])  # Initially the same as extracted_data
    card.logs = result['log']
    
    db.session.commit()
    
    return jsonify({
        'success': result['success'],
        'message': 'Information extracted successfully' if result['success'] else 'Failed to extract information',
        'data': result['data']
    })

@app.route('/card/<int:card_id>')
@login_required
def edit_card(card_id):
    card = BusinessCard.query.get_or_404(card_id)
    
    # Check if the card belongs to the current user or the user is an admin
    if card.user_id != current_user.id and not current_user.is_admin:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    extracted_data = json.loads(card.extracted_data) if card.extracted_data else {}
    print(extracted_data)
    edited_data = json.loads(card.edited_data) if card.edited_data else extracted_data
    print(edited_data)
    return render_template('edit_card.html', card=card, extracted_data=extracted_data, edited_data=edited_data)

@app.route('/save_card/<int:card_id>', methods=['POST'])
@login_required
def save_card(card_id):
    card = BusinessCard.query.get_or_404(card_id)
    
    # Check if the card belongs to the current user or the user is an admin
    if card.user_id != current_user.id and not current_user.is_admin:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get the edited data from the form
    # Get form data - arrays
    updated_data = {
        'Name': request.form.getlist('Name[]'),
        'Company': request.form.getlist('Company[]'),
        'Position': request.form.getlist('Position[]'),
        'Department': request.form.getlist('Department[]'), 
        'Phone': request.form.getlist('Phone[]'),
        'Email': request.form.getlist('Email[]'),
        'Address': request.form.getlist('Address[]'),
        'Other': request.form.getlist('Other[]')
    }
    print(updated_data)
    # Filter out empty values
    for key in updated_data:
        updated_data[key] = [value for value in updated_data[key] if value.strip()]
    
    # Update the card data
    card.edited_data = json.dumps(updated_data)
    db.session.commit()
    
    flash('Card information saved successfully', 'success')
    return redirect(url_for('dashboard'))

@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get all business cards
    cards = BusinessCard.query.order_by(BusinessCard.uploaded_at.desc()).all()
    
    # Get users count
    users_count = User.query.count()
    
    # Get cards count by status
    pending_count = BusinessCard.query.filter_by(status='pending').count()
    processed_count = BusinessCard.query.filter_by(status='processed').count()
    failed_count = BusinessCard.query.filter_by(status='failed').count()
    
    return render_template(
        'admin_dashboard.html', 
        cards=cards, 
        users_count=users_count, 
        pending_count=pending_count, 
        processed_count=processed_count, 
        failed_count=failed_count
    )

@app.route('/admin/edit_card/<int:card_id>')
@login_required
def admin_edit_card(card_id):
    if not current_user.is_admin:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    return redirect(url_for('edit_card', card_id=card_id))

@app.route('/admin/logs/<int:card_id>')
@login_required
def view_logs(card_id):
    if not current_user.is_admin:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    card = BusinessCard.query.get_or_404(card_id)
    
    return render_template('view_logs.html', card=card)

@app.route('/admin/export/<int:card_id>')
@login_required
def export_card(card_id):
    if not current_user.is_admin:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    card = BusinessCard.query.get_or_404(card_id)
    
    # Get card owner
    owner = User.query.get(card.user_id)
    
    # Prepare data for export
    edited_data = json.loads(card.edited_data) if card.edited_data else {}
    
    data = {
        'ID': card.id,
        'User': owner.username,
        'Original Filename': card.original_filename,
        'Status': card.status,
        'Uploaded At': card.uploaded_at,
        'Processed At': card.processed_at,
        'Name': edited_data.get('name', ''),
        'Position': edited_data.get('position', ''),
        'Company': edited_data.get('company', ''),
        'Email': edited_data.get('email', ''),
        'Phone': edited_data.get('phone', ''),
        'Website': edited_data.get('website', ''),
        'Address': edited_data.get('address', '')
    }
    
    # Create a DataFrame
    df = pd.DataFrame([data])
    
    # Define export path
    export_folder = 'static/exports'
    os.makedirs(export_folder, exist_ok=True)
    export_file = f"{export_folder}/card_{card.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Export to CSV
    df.to_csv(export_file, index=False)
    
    return redirect(url_for('download_file', filename=os.path.basename(export_file)))

@app.route('/download/<filename>')
@login_required
def download_file(filename):
    return send_from_directory('static/exports', filename, as_attachment=True)

@app.route('/user/export/<int:card_id>')
@login_required
def user_export_card(card_id):
    card = BusinessCard.query.get_or_404(card_id)
    
    # Check if the card belongs to the current user
    if card.user_id != current_user.id and not current_user.is_admin:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    # Prepare data for VCF export
    edited_data = json.loads(card.edited_data) if card.edited_data else {}
    
    # Create VCF content
    vcf_content = f"""BEGIN:VCARD
VERSION:3.0
FN:{edited_data.get('name', '')}
ORG:{edited_data.get('company', '')}
TITLE:{edited_data.get('position', '')}
TEL:{edited_data.get('phone', '')}
EMAIL:{edited_data.get('email', '')}
URL:{edited_data.get('website', '')}
ADR:{edited_data.get('address', '')}
END:VCARD
"""
    
    # Define export path
    export_folder = 'static/exports'
    os.makedirs(export_folder, exist_ok=True)
    export_file = f"{export_folder}/contact_{card.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.vcf"
    
    # Write VCF file
    with open(export_file, 'w') as f:
        f.write(vcf_content)
    
    return redirect(url_for('download_file', filename=os.path.basename(export_file)))

# Create all tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
