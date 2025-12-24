import os
import numpy as np
import pandas as pd
import uuid
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from flask import Flask, request, render_template_string, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from functools import wraps

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

app = Flask(__name__)
app.secret_key = "super_secret_key_epc_system_2025"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# Folder setup
DATASETS_FOLDER = 'datasets'
UPLOAD_FOLDER = 'static/uploads'
USERS_FILE = 'users.json'
LOGS_FILE = 'logs.json'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Create necessary directories
os.makedirs(DATASETS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# 2. USER AUTHENTICATION SYSTEM
# ==========================================

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    """Hash password with salt"""
    salt = secrets.token_hex(16)
    return salt + ':' + hashlib.sha256((salt + password).encode()).hexdigest()

def verify_password(stored_password, provided_password):
    """Verify hashed password"""
    salt, hashed = stored_password.split(':')
    return hashlib.sha256((salt + provided_password).encode()).hexdigest() == hashed

def init_users():
    """Initialize users file with admin user"""
    users = load_users()
    if not users:
        users['admin@epc.com'] = {
            'username': 'admin',
            'password': hash_password('admin123'),
            'full_name': 'System Administrator',
            'role': 'admin',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_login': None,
            'uploads_count': 0
        }
        save_users(users)
        print("✅ Admin user created: admin@epc.com / admin123")

def log_activity(user_email, action, details=""):
    """Log user activities"""
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user': user_email,
        'action': action,
        'details': details,
        'ip': request.remote_addr if request else 'N/A'
    }
    
    # Print to console (command prompt)
    print(f"📝 [{log_entry['timestamp']}] {user_email}: {action} {details}")
    
    # Save to file
    logs = []
    if os.path.exists(LOGS_FILE):
        try:
            with open(LOGS_FILE, 'r') as f:
                logs = json.load(f)
        except:
            pass
    
    logs.append(log_entry)
    
    # Keep only last 1000 logs
    if len(logs) > 1000:
        logs = logs[-1000:]
    
    with open(LOGS_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

# ==========================================
# 3. DECORATORS FOR AUTHENTICATION
# ==========================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            flash('⚠️ Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            flash('⚠️ Please login to access this page', 'warning')
            return redirect(url_for('login'))
        
        users = load_users()
        user_email = session['user_email']
        
        if user_email not in users or users[user_email].get('role') != 'admin':
            flash('🔒 Admin access required', 'danger')
            return redirect(url_for('index'))
            
        return f(*args, **kwargs)
    return decorated_function

# ==========================================
# 4. DATABASE & AI LOADING
# ==========================================

def load_product_database():
    """Load products from CSV files"""
    products_list = []
    
    if not os.path.exists(DATASETS_FOLDER):
        os.makedirs(DATASETS_FOLDER)
    
    csv_files = [f for f in os.listdir(DATASETS_FOLDER) if f.endswith('.csv')]
    
    if not csv_files:
        print("⚠️ No CSV files found in datasets folder")
        return []

    for filename in csv_files:
        store_name = os.path.splitext(filename)[0].title()
        file_path = os.path.join(DATASETS_FOLDER, filename)
        
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower().str.strip()
            
            for _, row in df.iterrows():
                if 'name' not in row or 'price' not in row:
                    continue
                
                products_list.append({
                    "id": len(products_list) + 1,
                    "name": str(row['name']).strip(),
                    "category": str(row.get('category', 'unknown')).lower(),
                    "store": store_name,
                    "price": float(row['price']),
                    "currency": "₹",
                    "url": str(row.get('url', '#')),
                    "rating": row.get('rating', '4.0'),
                    "brand": str(row.get('brand', 'Generic')).title(),
                    "model_id": str(row.get('model_id', 'N/A')),
                    "stock": str(row.get('stock', 'In Stock')),
                    "discount_percent": row.get('discount_percent', 0),
                    "description": str(row.get('description', 'No description available.')),
                    "image_url": str(row.get('image_url', ''))
                })
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    print(f"✅ Loaded {len(products_list)} products from {len(csv_files)} stores")
    return products_list

# Global variables
PRODUCT_DATABASE = load_product_database()
CSV_FILES = [f for f in os.listdir(DATASETS_FOLDER) if f.endswith('.csv')] if os.path.exists(DATASETS_FOLDER) else []

print("🤖 Loading AI Model...")
try:
    model = MobileNetV2(weights='imagenet')
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def prepare_image(image, target_size=(224, 224)):
    """Prepare image for model prediction"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# ==========================================
# 5. HTML TEMPLATES
# ==========================================

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en" data-bs-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EPC | Smart Price Comparison</title>
    
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Bootstrap Icons -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
    
    <!-- Animate.css -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">
    
    <!-- AOS Animation -->
    <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
    
    <style>
        :root {
            --primary-color: #6366f1;
            --primary-dark: #4f46e5;
            --secondary-color: #10b981;
            --accent-color: #f59e0b;
            --danger-color: #ef4444;
            --dark-color: #1e293b;
            --light-color: #f8fafc;
            --gray-color: #64748b;
            --card-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            --hover-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.15);
        }
        
        [data-bs-theme="dark"] {
            --primary-color: #818cf8;
            --primary-dark: #6366f1;
            --dark-color: #0f172a;
            --light-color: #1e293b;
            --card-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: var(--dark-color);
            overflow-x: hidden;
        }
        
        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 50px;
            padding: 10px 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            display: flex;
            align-items: center;
            gap: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .theme-toggle:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        }
        
        .glass-effect {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: var(--card-shadow);
        }
        
        [data-bs-theme="dark"] .glass-effect {
            background: rgba(30, 41, 59, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1.5rem;
        }
        
        @media (max-width: 768px) {
            .hero-title {
                font-size: 2.5rem;
            }
        }
        
        .feature-card {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            height: 100%;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            position: relative;
            overflow: hidden;
        }
        
        [data-bs-theme="dark"] .feature-card {
            background: var(--light-color);
            color: white;
        }
        
        .feature-card:hover {
            transform: translateY(-10px);
            box-shadow: var(--hover-shadow);
            border-color: var(--primary-color);
        }
        
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        }
        
        .feature-icon {
            width: 70px;
            height: 70px;
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.5rem;
            color: white;
            font-size: 1.8rem;
        }
        
        .upload-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 24px;
            padding: 3rem;
            margin: 3rem auto;
            max-width: 800px;
        }
        
        [data-bs-theme="dark"] .upload-container {
            background: rgba(30, 41, 59, 0.95);
        }
        
        .upload-area {
            border: 3px dashed var(--primary-color);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(99, 102, 241, 0.05);
            position: relative;
            overflow: hidden;
        }
        
        .upload-area:hover {
            border-color: var(--primary-dark);
            background: rgba(99, 102, 241, 0.1);
            transform: scale(1.01);
        }
        
        .upload-area.active {
            border-color: var(--secondary-color);
            background: rgba(16, 185, 129, 0.1);
        }
        
        .upload-icon {
            font-size: 4rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover .upload-icon {
            transform: scale(1.1) rotate(5deg);
        }
        
        .product-card {
            background: white;
            border-radius: 16px;
            overflow: hidden;
            transition: all 0.3s ease;
            height: 100%;
            border: 1px solid #e2e8f0;
            position: relative;
        }
        
        [data-bs-theme="dark"] .product-card {
            background: var(--light-color);
            border-color: #334155;
        }
        
        .product-card:hover {
            transform: translateY(-8px);
            box-shadow: var(--hover-shadow);
            border-color: var(--primary-color);
        }
        
        .product-badge {
            position: absolute;
            top: 15px;
            left: 15px;
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-size: 0.8rem;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
            z-index: 1;
        }
        
        .best-price-badge {
            position: absolute;
            top: 15px;
            right: 15px;
            background: linear-gradient(135deg, var(--secondary-color), #059669);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-size: 0.8rem;
            font-weight: 600;
            animation: pulse 2s infinite;
            z-index: 1;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        
        .price-tag {
            font-size: 2rem;
            font-weight: 800;
            color: var(--primary-color);
            margin: 1rem 0;
        }
        
        .discount-badge {
            background: linear-gradient(135deg, var(--danger-color), #dc2626);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }
        
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.95);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            backdrop-filter: blur(5px);
        }
        
        [data-bs-theme="dark"] .loading-overlay {
            background: rgba(15, 23, 42, 0.95);
        }
        
        .loader {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(99, 102, 241, 0.1);
            border-top-color: var(--primary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .notification-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            max-width: 400px;
        }
        
        .notification {
            background: white;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: var(--card-shadow);
            border-left: 5px solid var(--primary-color);
            display: flex;
            align-items: center;
            gap: 12px;
            animation: slideInRight 0.3s ease;
            transform-origin: right;
        }
        
        [data-bs-theme="dark"] .notification {
            background: var(--light-color);
            color: white;
        }
        
        .notification.success {
            border-left-color: var(--secondary-color);
        }
        
        .notification.warning {
            border-left-color: var(--accent-color);
        }
        
        .notification.danger {
            border-left-color: var(--danger-color);
        }
        
        .notification-icon {
            font-size: 1.5rem;
        }
        
        @keyframes slideInRight {
            from { transform: translateX(100%) scale(0.8); opacity: 0; }
            to { transform: translateX(0) scale(1); opacity: 1; }
        }
        
        @keyframes slideOutRight {
            from { transform: translateX(0) scale(1); opacity: 1; }
            to { transform: translateX(100%) scale(0.8); opacity: 0; }
        }
        
        .stats-card {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            color: white;
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .stats-card::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        }
        
        .stats-number {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            display: block;
        }
        
        .btn-gradient {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            color: white;
            border: none;
            border-radius: 50px;
            padding: 0.875rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .btn-gradient::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s ease;
        }
        
        .btn-gradient:hover::before {
            left: 100%;
        }
        
        .btn-gradient:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(99, 102, 241, 0.3);
        }
        
        .fab {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
            cursor: pointer;
            z-index: 100;
            transition: all 0.3s ease;
        }
        
        .fab:hover {
            transform: scale(1.1) rotate(90deg);
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.6);
        }
        
        @media (max-width: 768px) {
            .container {
                padding-left: 15px;
                padding-right: 15px;
            }
            
            .upload-container {
                padding: 2rem 1rem;
                margin: 1rem auto;
            }
            
            .stats-number {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <!-- Theme Toggle -->
    <div class="theme-toggle" onclick="toggleTheme()">
        <i class="bi bi-sun-fill theme-icon"></i>
        <span>Light Mode</span>
    </div>
    
    <!-- Loading Overlay -->
    <div id="loadingOverlay" class="loading-overlay" style="display: none;">
        <div class="loader mb-3"></div>
        <p class="text-muted">Processing your request...</p>
    </div>
    
    <!-- Notifications Container -->
    <div class="notification-container">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="notification {{ category }} animate__animated animate__fadeInRight">
                        <i class="fas 
                            {% if category == 'success' %}fa-check-circle text-success
                            {% elif category == 'warning' %}fa-exclamation-triangle text-warning
                            {% elif category == 'danger' %}fa-times-circle text-danger
                            {% else %}fa-info-circle text-primary{% endif %}
                            notification-icon">
                        </i>
                        <div>
                            <strong>{{ category|title }}:</strong>
                            <span>{{ message }}</span>
                        </div>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
    </div>
    
    <!-- Floating Action Button -->
    <div class="fab" onclick="scrollToTop()">
        <i class="bi bi-arrow-up"></i>
    </div>
    
    {% block main_content %}{% endblock %}
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- AOS Animation -->
    <script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
    
    <!-- Chart.js for statistics -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <script>
        // Initialize AOS
        AOS.init({
            duration: 1000,
            once: true
        });
        
        // Theme Toggle
        function toggleTheme() {
            const html = document.documentElement;
            const themeIcon = document.querySelector('.theme-icon');
            const themeText = document.querySelector('.theme-toggle span');
            
            if (html.getAttribute('data-bs-theme') === 'dark') {
                html.setAttribute('data-bs-theme', 'light');
                themeIcon.className = 'bi bi-sun-fill theme-icon';
                themeText.textContent = 'Light Mode';
                localStorage.setItem('theme', 'light');
            } else {
                html.setAttribute('data-bs-theme', 'dark');
                themeIcon.className = 'bi bi-moon-fill theme-icon';
                themeText.textContent = 'Dark Mode';
                localStorage.setItem('theme', 'dark');
            }
        }
        
        // Load saved theme
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-bs-theme', savedTheme);
        const themeIcon = document.querySelector('.theme-icon');
        const themeText = document.querySelector('.theme-toggle span');
        
        if (savedTheme === 'dark') {
            themeIcon.className = 'bi bi-moon-fill theme-icon';
            themeText.textContent = 'Dark Mode';
        }
        
        // Scroll to top
        function scrollToTop() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        }
        
        // Show/hide FAB based on scroll
        window.addEventListener('scroll', function() {
            const fab = document.querySelector('.fab');
            if (window.scrollY > 300) {
                fab.style.display = 'flex';
            } else {
                fab.style.display = 'none';
            }
        });
        
        // Auto-hide notifications
        setTimeout(() => {
            const notifications = document.querySelectorAll('.notification');
            notifications.forEach(notification => {
                notification.style.animation = 'slideOutRight 0.3s ease';
                setTimeout(() => notification.remove(), 300);
            });
        }, 5000);
        
        // Show loading
        function showLoading() {
            document.getElementById('loadingOverlay').style.display = 'flex';
        }
        
        // Hide loading
        function hideLoading() {
            document.getElementById('loadingOverlay').style.display = 'none';
        }
        
        // Initialize tooltips
        document.addEventListener('DOMContentLoaded', function() {
            const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
            tooltips.forEach(tooltip => new bootstrap.Tooltip(tooltip));
        });
        
        // Form submission with loading
        document.addEventListener('DOMContentLoaded', function() {
            const forms = document.querySelectorAll('form');
            forms.forEach(form => {
                form.addEventListener('submit', function(e) {
                    if (!this.classList.contains('no-loading')) {
                        showLoading();
                    }
                });
            });
        });
    </script>
    
    {% block scripts %}{% endblock %}
</body>
</html>
'''

# ==========================================
# 6. ROUTES - AUTHENTICATION
# ==========================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        username = request.form.get('username', '').strip()
        full_name = request.form.get('full_name', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        if not all([email, username, full_name, password]):
            flash('All fields are required', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'danger')
            return redirect(url_for('register'))
        
        # Load users
        users = load_users()
        
        # Check if user exists
        if email in users:
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        users[email] = {
            'username': username,
            'password': hash_password(password),
            'full_name': full_name,
            'role': 'user',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_login': None,
            'uploads_count': 0,
            'searches': []
        }
        
        # Save users
        save_users(users)
        
        # Log activity
        log_activity(email, 'REGISTER', f'New user: {username}')
        
        flash('Registration successful! Please login', 'success')
        return redirect(url_for('login'))
    
    # GET request - show registration form
    return render_template_string('''
{% extends "base.html" %}
{% block main_content %}
<div class="auth-container animate__animated animate__fadeIn">
    <div class="auth-card">
        <div class="auth-header">
            <div class="auth-logo">
                <i class="fas fa-balance-scale-left"></i>
            </div>
            <h2 class="auth-title">Create Account</h2>
            <p class="auth-subtitle">Join our price comparison community</p>
        </div>
        
        <form method="POST" action="/register">
            <div class="form-group">
                <label class="form-label">Full Name</label>
                <input type="text" class="form-control" name="full_name" required 
                       placeholder="Enter your full name">
            </div>
            
            <div class="form-group">
                <label class="form-label">Username</label>
                <input type="text" class="form-control" name="username" required 
                       placeholder="Choose a username">
            </div>
            
            <div class="form-group">
                <label class="form-label">Email Address</label>
                <input type="email" class="form-control" name="email" required 
                       placeholder="Enter your email">
            </div>
            
            <div class="form-group">
                <label class="form-label">Password</label>
                <input type="password" class="form-control" name="password" required 
                       placeholder="Create a password (min. 6 characters)">
            </div>
            
            <div class="form-group">
                <label class="form-label">Confirm Password</label>
                <input type="password" class="form-control" name="confirm_password" required 
                       placeholder="Confirm your password">
            </div>
            
            <button type="submit" class="btn-auth">
                <i class="fas fa-user-plus me-2"></i>Create Account
            </button>
        </form>
        
        <div class="auth-footer">
            <p>Already have an account? <a href="/login" class="auth-link">Sign In</a></p>
            <p><a href="/" class="auth-link"><i class="fas fa-arrow-left me-1"></i> Back to Home</a></p>
        </div>
    </div>
</div>
{% endblock %}
''')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        # Validation
        if not email or not password:
            flash('Email and password are required', 'danger')
            return redirect(url_for('login'))
        
        # Load users
        users = load_users()
        
        # Check if user exists
        if email not in users:
            flash('Invalid email or password', 'danger')
            return redirect(url_for('login'))
        
        # Verify password
        if not verify_password(users[email]['password'], password):
            flash('Invalid email or password', 'danger')
            return redirect(url_for('login'))
        
        # Update last login
        users[email]['last_login'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        save_users(users)
        
        # Set session
        session['user_email'] = email
        session['user_name'] = users[email]['username']
        session['full_name'] = users[email]['full_name']
        session['role'] = users[email]['role']
        session.permanent = True
        
        # Log activity
        log_activity(email, 'LOGIN', f'Successful login from {request.remote_addr}')
        
        flash(f'Welcome back, {users[email]["username"]}!', 'success')
        
        # Redirect based on role
        if users[email]['role'] == 'admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('index'))
    
    # GET request - show login form
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - EPC System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
        }
        
        .login-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 24px;
            padding: 3rem;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .login-header {
            text-align: center;
            margin-bottom: 2.5rem;
        }
        
        .login-logo {
            font-size: 3rem;
            color: #6366f1;
            margin-bottom: 1rem;
        }
        
        .form-control {
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.875rem 1rem;
            transition: all 0.3s ease;
        }
        
        .form-control:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        
        .btn-login {
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .btn-login:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(99, 102, 241, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-6 col-lg-5">
                <div class="login-card">
                    <div class="login-header">
                        <div class="login-logo">
                            <i class="bi bi-graph-up-arrow"></i>
                        </div>
                        <h2 class="fw-bold mb-2">Welcome Back</h2>
                        <p class="text-muted">Sign in to your EPC account</p>
                    </div>
                    
                    <form method="POST" action="/login">
                        <div class="mb-3">
                            <label class="form-label fw-bold">Email Address</label>
                            <input type="email" class="form-control" name="email" 
                                   placeholder="Enter your email" required>
                        </div>
                        
                        <div class="mb-4">
                            <label class="form-label fw-bold">Password</label>
                            <input type="password" class="form-control" name="password" 
                                   placeholder="Enter your password" required>
                        </div>
                        
                        <div class="d-flex justify-content-between align-items-center mb-4">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="remember">
                                <label class="form-check-label" for="remember">
                                    Remember me
                                </label>
                            </div>
                            <a href="#" class="text-decoration-none">Forgot password?</a>
                        </div>
                        
                        <button type="submit" class="btn-login w-100 mb-3">
                            <i class="bi bi-box-arrow-in-right me-2"></i>Sign In
                        </button>
                        
                        <div class="text-center">
                            <p class="text-muted mb-0">
                                Don't have an account? 
                                <a href="/register" class="text-decoration-none fw-bold">
                                    Sign up now
                                </a>
                            </p>
                            <a href="/" class="text-decoration-none mt-2 d-block">
                                <i class="bi bi-arrow-left me-1"></i>Back to Home
                            </a>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
''')

@app.route('/logout')
def logout():
    """User logout"""
    if 'user_email' in session:
        log_activity(session['user_email'], 'LOGOUT', 'User logged out')
        session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

# ==========================================
# 7. MAIN HOME PAGE
# ==========================================

@app.route('/')
def index():
    """Main home page"""
    users = load_users()
    total_users = len(users)
    active_users = sum(1 for u in users.values() if u.get('last_login'))
    
    # Get any passed parameters
    prediction = request.args.get('prediction', '')
    matches_str = request.args.get('matches', '[]')
    uploaded_image_url = request.args.get('uploaded_image_url', '')
    global_best_price = request.args.get('global_best_price', '')
    
    try:
        matches = eval(matches_str)
    except:
        matches = []
    
    template = BASE_TEMPLATE.replace('{% block main_content %}{% endblock %}', '''
{% block main_content %}
<!-- Navbar -->
<nav class="navbar navbar-expand-lg navbar-dark fixed-top" style="background: rgba(30, 41, 59, 0.95); backdrop-filter: blur(10px);">
    <div class="container">
        <a class="navbar-brand d-flex align-items-center" href="/" style="font-weight: 800; font-size: 1.5rem;">
            <i class="bi bi-graph-up-arrow me-2" style="color: #6366f1;"></i>
            <span style="background: linear-gradient(135deg, #6366f1, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                EPC
            </span>
        </a>
        
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
            <span class="navbar-toggler-icon"></span>
        </button>
        
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav me-auto">
                <li class="nav-item">
                    <a class="nav-link active" href="/">
                        <i class="bi bi-house-door me-1"></i> Home
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#features">
                        <i class="bi bi-stars me-1"></i> Features
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#upload">
                        <i class="bi bi-search me-1"></i> Compare
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#how-it-works">
                        <i class="bi bi-info-circle me-1"></i> How It Works
                    </a>
                </li>
                ''' + ('''
                <li class="nav-item">
                    <a class="nav-link" href="/admin/dashboard">
                        <i class="bi bi-speedometer2 me-1"></i> Admin
                    </a>
                </li>
                ''' if 'user_email' in session and session.get('role') == 'admin' else '') + '''
            </ul>
            
            <div class="navbar-nav ms-auto">
                ''' + ('''
                <div class="dropdown">
                    <a href="#" class="nav-link dropdown-toggle d-flex align-items-center" data-bs-toggle="dropdown">
                        <div class="user-avatar me-2" style="width: 32px; height: 32px; background: linear-gradient(135deg, #6366f1, #10b981); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600;">
                            ''' + (session.get('user_name', 'U')[0].upper() if session.get('user_name') else 'U') + '''
                        </div>
                        ''' + (session.get('user_name', 'User') if session.get('user_name') else 'User') + '''
                    </a>
                    <ul class="dropdown-menu dropdown-menu-end">
                        <li>
                            <a class="dropdown-item" href="/profile">
                                <i class="bi bi-person me-2"></i> Profile
                            </a>
                        </li>
                        <li>
                            <a class="dropdown-item" href="/my-searches">
                                <i class="bi bi-clock-history me-2"></i> Search History
                            </a>
                        </li>
                        <li><hr class="dropdown-divider"></li>
                        <li>
                            <a class="dropdown-item" href="/logout">
                                <i class="bi bi-box-arrow-right me-2"></i> Logout
                            </a>
                        </li>
                    </ul>
                </div>
                ''' if 'user_email' in session else '''
                <a href="/login" class="btn btn-outline-light me-2">
                    <i class="bi bi-box-arrow-in-right me-1"></i> Login
                </a>
                <a href="/register" class="btn btn-gradient">
                    <i class="bi bi-person-plus me-1"></i> Register
                </a>
                ''') + '''
            </div>
        </div>
    </div>
</nav>

<!-- Hero Section -->
<section class="hero-section">
    <div class="hero-bg"></div>
    <div class="container">
        <div class="row align-items-center">
            <div class="col-lg-6" data-aos="fade-right">
                <h1 class="hero-title animate__animated animate__fadeInUp">
                    Smart Price Comparison Made Simple
                </h1>
                <p class="lead text-dark mb-4 animate__animated animate__fadeInUp animate__delay-1s" style="font-size: 1.25rem;">
                    Upload product images or search by keywords to find the best prices across 
                    multiple e-commerce stores using our advanced AI-powered recognition system.
                </p>
                <div class="d-flex flex-wrap gap-3 animate__animated animate__fadeInUp animate__delay-2s">
                    <a href="#upload" class="btn btn-gradient btn-lg">
                        <i class="bi bi-cloud-upload me-2"></i> Start Comparing
                    </a>
                    <a href="#how-it-works" class="btn btn-outline-dark btn-lg">
                        <i class="bi bi-play-circle me-2"></i> Watch Demo
                    </a>
                </div>
                
                <div class="mt-5">
                    <div class="row">
                        <div class="col-4">
                            <div class="text-center">
                                <h3 class="text-primary fw-bold">''' + str(len(PRODUCT_DATABASE)) + '''</h3>
                                <p class="text-muted mb-0">Products</p>
                            </div>
                        </div>
                        <div class="col-4">
                            <div class="text-center">
                                <h3 class="text-primary fw-bold">''' + str(len(CSV_FILES)) + '''</h3>
                                <p class="text-muted mb-0">Stores</p>
                            </div>
                        </div>
                        <div class="col-4">
                            <div class="text-center">
                                <h3 class="text-primary fw-bold">''' + str(total_users) + '''</h3>
                                <p class="text-muted mb-0">Users</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-lg-6" data-aos="fade-left">
                <div class="glass-effect rounded-4 p-4 shadow-lg">
                    <div class="text-center mb-4">
                        <i class="bi bi-robot display-4 text-primary"></i>
                        <h4 class="mt-3">AI-Powered Detection</h4>
                        <p class="text-muted">Upload any product image for instant recognition</p>
                    </div>
                    
                    <div class="upload-area rounded-3 p-5 mb-4" id="uploadArea">
                        <i class="bi bi-cloud-arrow-up upload-icon"></i>
                        <h5>Drag & Drop Image</h5>
                        <p class="text-muted mb-0">or click to browse (JPG, PNG, WebP)</p>
                        <small class="text-muted">Max file size: 16MB</small>
                    </div>
                    
                    <div class="d-grid">
                        <button class="btn btn-gradient" onclick="document.getElementById('fileInput').click()">
                            <i class="bi bi-camera me-2"></i> Upload Product Image
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</section>

<!-- Features Section -->
<section id="features" class="py-5">
    <div class="container">
        <div class="text-center mb-5" data-aos="fade-up">
            <h2 class="display-5 fw-bold mb-3">Why Choose EPC?</h2>
            <p class="lead text-muted">Experience the future of price comparison</p>
        </div>
        
        <div class="row g-4">
            <div class="col-md-4" data-aos="fade-up" data-aos-delay="100">
                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="bi bi-robot"></i>
                    </div>
                    <h4 class="mb-3">Advanced AI Recognition</h4>
                    <p class="text-muted">
                        Our state-of-the-art MobileNetV2 AI model identifies products from images 
                        with 95% accuracy, trained on millions of product images.
                    </p>
                </div>
            </div>
            
            <div class="col-md-4" data-aos="fade-up" data-aos-delay="200">
                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="bi bi-lightning-charge"></i>
                    </div>
                    <h4 class="mb-3">Real-Time Comparison</h4>
                    <p class="text-muted">
                        Get instant price comparisons across all major e-commerce platforms. 
                        Our database updates continuously with live prices.
                    </p>
                </div>
            </div>
            
            <div class="col-md-4" data-aos="fade-up" data-aos-delay="300">
                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="bi bi-shield-check"></i>
                    </div>
                    <h4 class="mb-3">Secure & Private</h4>
                    <p class="text-muted">
                        Your data is encrypted end-to-end. We never share your information 
                        and ensure complete privacy protection.
                    </p>
                </div>
            </div>
        </div>
    </div>
</section>

<!-- Upload Section -->
<section id="upload" class="py-5" style="background: rgba(255, 255, 255, 0.02);">
    <div class="container">
        <div class="upload-container" data-aos="fade-up">
            <div class="text-center mb-5">
                <h2 class="fw-bold mb-3">Find The Best Prices</h2>
                <p class="text-muted">Upload an image or search by keywords to compare prices</p>
            </div>
            
            ''' + ('''
            <div class="alert alert-warning alert-dismissible fade show mb-4" role="alert">
                <i class="bi bi-info-circle me-2"></i>
                <a href="/login" class="alert-link">Login</a> or 
                <a href="/register" class="alert-link">Register</a> to save your search history and preferences!
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
            ''' if 'user_email' not in session else '') + '''
            
            <form action="/process" method="post" enctype="multipart/form-data" id="mainForm">
                <!-- File Upload -->
                <div class="mb-4">
                    <label class="form-label fw-bold">
                        <i class="bi bi-camera me-2"></i>Upload Product Image
                    </label>
                    
                    <div class="upload-area" onclick="document.getElementById('fileInput').click()" 
                         id="uploadArea">
                        <i class="bi bi-cloud-arrow-up upload-icon"></i>
                        <h5>Click to upload image</h5>
                        <p class="text-muted mb-0">
                            Drag & drop or click to browse (JPG, PNG, WebP)
                        </p>
                        <small class="text-muted">Max file size: 16MB</small>
                    </div>
                    
                    <input type="file" id="fileInput" name="file" class="d-none" 
                           accept="image/*" onchange="handleFileUpload(this)" required>
                    
                    <div class="upload-progress" id="uploadProgress">
                        <div class="upload-progress-bar" id="progressBar"></div>
                    </div>
                    
                    <div id="imagePreview" class="mt-3"></div>
                </div>
                
                <!-- Keywords Section -->
                <div class="mb-4">
                    <label class="form-label fw-bold">
                        <i class="bi bi-tags me-2"></i>Search Keywords (Optional)
                    </label>
                    <div class="input-group">
                        <span class="input-group-text">
                            <i class="bi bi-search"></i>
                        </span>
                        <input type="text" class="form-control form-control-lg" 
                               name="keywords" 
                               placeholder="e.g., iPhone 15, Nike shoes, wireless headphones..."
                               id="keywordsInput">
                    </div>
                    <div class="mt-2">
                        <div class="form-text">
                            Add specific keywords to refine your search results
                        </div>
                    </div>
                </div>
                
                <!-- Action Buttons -->
                <div class="d-grid gap-3">
                    <button type="submit" class="btn btn-gradient btn-lg">
                        <i class="bi bi-search me-2"></i> Search & Compare Prices
                    </button>
                    
                    <div class="d-flex justify-content-between">
                        <a href="/reload_csv" class="btn btn-outline-secondary">
                            <i class="bi bi-arrow-clockwise me-1"></i> Refresh Database
                        </a>
                        <button type="button" class="btn btn-outline-primary" 
                                onclick="clearForm()">
                            <i class="bi bi-trash me-1"></i> Clear Form
                        </button>
                    </div>
                </div>
            </form>
        </div>
    </div>
</section>

<!-- How It Works -->
<section id="how-it-works" class="py-5">
    <div class="container">
        <div class="text-center mb-5" data-aos="fade-up">
            <h2 class="display-5 fw-bold mb-3">How It Works</h2>
            <p class="lead text-muted">Four simple steps to find the best prices</p>
        </div>
        
        <div class="row">
            <div class="col-lg-3 col-md-6 mb-4" data-aos="fade-up" data-aos-delay="100">
                <div class="text-center p-4">
                    <div class="feature-icon mx-auto mb-4">
                        <i class="bi bi-cloud-upload"></i>
                    </div>
                    <h4 class="mb-3">1. Upload Image</h4>
                    <p class="text-muted">Take or upload a clear photo of the product</p>
                </div>
            </div>
            
            <div class="col-lg-3 col-md-6 mb-4" data-aos="fade-up" data-aos-delay="200">
                <div class="text-center p-4">
                    <div class="feature-icon mx-auto mb-4">
                        <i class="bi bi-cpu"></i>
                    </div>
                    <h4 class="mb-3">2. AI Analysis</h4>
                    <p class="text-muted">Our AI identifies product details and category</p>
                </div>
            </div>
            
            <div class="col-lg-3 col-md-6 mb-4" data-aos="fade-up" data-aos-delay="300">
                <div class="text-center p-4">
                    <div class="feature-icon mx-auto mb-4">
                        <i class="bi bi-search-heart"></i>
                    </div>
                    <h4 class="mb-3">3. Smart Search</h4>
                    <p class="text-muted">Search across millions of products instantly</p>
                </div>
            </div>
            
            <div class="col-lg-3 col-md-6 mb-4" data-aos="fade-up" data-aos-delay="400">
                <div class="text-center p-4">
                    <div class="feature-icon mx-auto mb-4">
                        <i class="bi bi-trophy"></i>
                    </div>
                    <h4 class="mb-3">4. Best Deal</h4>
                    <p class="text-muted">Compare prices and get the best offer</p>
                </div>
            </div>
        </div>
    </div>
</section>

<!-- Results Section -->
''' + ('''
<section class="py-5">
    <div class="container">
        <div class="upload-container" data-aos="fade-up">
            <div class="d-flex flex-column flex-md-row justify-content-between align-items-start align-items-md-center mb-4">
                <div>
                    <h4 class="mb-2">
                        <i class="bi bi-robot text-primary me-2"></i>AI Detection Result
                    </h4>
                    <div class="badge bg-primary bg-gradient fs-6 p-2">
                        ''' + prediction + '''
                    </div>
                </div>
                ''' + ('''
                <div class="badge bg-success bg-gradient fs-6 p-2 mt-2 mt-md-0">
                    ''' + str(len(matches)) + ''' products found
                </div>
                ''' if matches else '') + '''
            </div>
            
            ''' + ('''
            <div class="alert alert-info">
                <i class="bi bi-exclamation-triangle me-2"></i>
                No exact matches found. Try using different keywords or upload a clearer image.
            </div>
            ''' if not matches else '') + '''
        </div>
    </div>
</section>
''' if prediction else '') + '''

<!-- Products Grid -->
''' + ('''
<section class="py-5">
    <div class="container">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h3 class="fw-bold">Best Deals Found</h3>
            <div class="dropdown">
                <button class="btn btn-outline-secondary dropdown-toggle" type="button" 
                        data-bs-toggle="dropdown">
                    <i class="bi bi-filter me-2"></i>Filter
                </button>
                <ul class="dropdown-menu">
                    <li><a class="dropdown-item" href="#">Price: Low to High</a></li>
                    <li><a class="dropdown-item" href="#">Price: High to Low</a></li>
                    <li><a class="dropdown-item" href="#">Best Rating</a></li>
                    <li><a class="dropdown-item" href="#">Highest Discount</a></li>
                </ul>
            </div>
        </div>
        
        <div class="row g-4">
''' + ''.join([f'''
            <div class="col-xl-3 col-lg-4 col-md-6" data-aos="fade-up">
                <div class="product-card">
                    <div class="position-relative">
                        <img src="{uploaded_image_url if uploaded_image_url else 'https://via.placeholder.com/300x250/e2e8f0/64748b?text=Product'}" 
                             class="product-image" alt="{match['name']}">
                        <span class="product-badge">{match['store']}</span>
                        ''' + (f'''
                        <span class="best-price-badge">Best Price</span>
                        ''' if global_best_price and match['price'] == float(global_best_price) else '') + '''
                        ''' + (f'''
                        <span class="badge bg-danger position-absolute" 
                              style="bottom: 15px; left: 15px;">
                            -{match['discount_percent']}% OFF
                        </span>
                        ''' if float(match.get('discount_percent', 0)) > 20 else '') + '''
                    </div>
                    
                    <div class="p-3">
                        <h5 class="mb-2" style="height: 3rem; overflow: hidden;">
                            {match['name']}
                        </h5>
                        
                        <div class="d-flex align-items-center mb-2">
                            <span class="price-tag">
                                {match['currency']}{"{:.2f}".format(match['price'])}
                            </span>
                            ''' + (f'''
                            <span class="discount-badge">Save {match['discount_percent']}%</span>
                            ''' if float(match.get('discount_percent', 0)) > 0 else '') + '''
                        </div>
                        
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <div class="d-flex align-items-center">
                                <i class="bi bi-star-fill text-warning me-1"></i>
                                <span>{match['rating']}</span>
                            </div>
                            <span class="badge bg-light text-dark">
                                <i class="bi bi-tag me-1"></i>{match['brand']}
                            </span>
                        </div>
                        
                        <p class="small text-muted mb-3" style="height: 3rem; overflow: hidden;">
                            {match['description'][:80]}{'...' if len(match['description']) > 80 else ''}
                        </p>
                        
                        <div class="d-grid gap-2">
                            <a href="{match['url']}" target="_blank" 
                               class="btn btn-outline-primary">
                                <i class="bi bi-cart-plus me-2"></i>View on {match['store']}
                            </a>
                            <button class="btn btn-light" 
                                    onclick="addToWishlist('{match['id']}')">
                                <i class="bi bi-heart me-2"></i>Save for Later
                            </button>
                        </div>
                    </div>
                </div>
            </div>
''' for match in matches]) + '''
        </div>
    </div>
</section>
''' if matches else '') + '''

<!-- Footer -->
<footer class="footer mt-5 py-5" style="background: rgba(30, 41, 59, 0.95); backdrop-filter: blur(10px); border-top: 1px solid rgba(255, 255, 255, 0.1);">
    <div class="container">
        <div class="row">
            <div class="col-lg-4 mb-4">
                <h5 class="text-white mb-4">
                    <i class="bi bi-graph-up-arrow me-2" style="color: #6366f1;"></i>
                    EPC System
                </h5>
                <p class="text-muted">
                    Advanced AI-powered price comparison system that helps you find the best deals across multiple e-commerce platforms.
                </p>
                <div class="social-links mt-3">
                    <a href="#" class="text-muted me-3"><i class="bi bi-twitter"></i></a>
                    <a href="#" class="text-muted me-3"><i class="bi bi-facebook"></i></a>
                    <a href="#" class="text-muted me-3"><i class="bi bi-instagram"></i></a>
                    <a href="#" class="text-muted"><i class="bi bi-linkedin"></i></a>
                </div>
            </div>
            
            <div class="col-lg-2 col-6 mb-4">
                <h6 class="text-white mb-4">Quick Links</h6>
                <ul class="list-unstyled">
                    <li class="mb-2"><a href="/" class="text-muted text-decoration-none">Home</a></li>
                    <li class="mb-2"><a href="#features" class="text-muted text-decoration-none">Features</a></li>
                    <li class="mb-2"><a href="#upload" class="text-muted text-decoration-none">Search</a></li>
                    <li class="mb-2"><a href="#how-it-works" class="text-muted text-decoration-none">How It Works</a></li>
                </ul>
            </div>
            
            <div class="col-lg-2 col-6 mb-4">
                <h6 class="text-white mb-4">Account</h6>
                <ul class="list-unstyled">
                    ''' + ('''
                    <li class="mb-2"><a href="/profile" class="text-muted text-decoration-none">Profile</a></li>
                    <li class="mb-2"><a href="/my-searches" class="text-muted text-decoration-none">History</a></li>
                    <li class="mb-2"><a href="/logout" class="text-muted text-decoration-none">Logout</a></li>
                    ''' if 'user_email' in session else '''
                    <li class="mb-2"><a href="/login" class="text-muted text-decoration-none">Login</a></li>
                    <li class="mb-2"><a href="/register" class="text-muted text-decoration-none">Register</a></li>
                    ''') + '''
                </ul>
            </div>
            
            <div class="col-lg-4 mb-4">
                <h6 class="text-white mb-4">Contact Info</h6>
                <ul class="list-unstyled text-muted">
                    <li class="mb-2"><i class="bi bi-envelope me-2"></i> support@epc.com</li>
                    <li class="mb-2"><i class="bi bi-telephone me-2"></i> +1 (555) 123-4567</li>
                    <li><i class="bi bi-geo-alt me-2"></i> 123 Tech Street, Silicon Valley</li>
                </ul>
            </div>
        </div>
        
        <hr style="border-color: rgba(255, 255, 255, 0.1);">
        
        <div class="row">
            <div class="col-md-6">
                <p class="text-muted mb-0">
                    &copy; 2024 EPC System. All rights reserved.
                </p>
            </div>
            <div class="col-md-6 text-md-end">
                <a href="#" class="text-muted text-decoration-none me-3">Privacy Policy</a>
                <a href="#" class="text-muted text-decoration-none">Terms of Service</a>
            </div>
        </div>
    </div>
</footer>
{% endblock %}

<script>
    // File upload with preview
    function handleFileUpload(input) {
        const file = input.files[0];
        if (file) {
            if (file.size > 16 * 1024 * 1024) {
                alert("File size must be less than 16MB");
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                const preview = document.getElementById('imagePreview');
                preview.innerHTML = `
                    <div class="text-center">
                        <img src="${e.target.result}" class="img-fluid rounded" style="max-height: 200px;">
                        <div class="mt-3">
                            <button type="button" class="btn btn-sm btn-outline-danger" onclick="clearImage()">
                                <i class="bi bi-trash"></i> Remove
                            </button>
                        </div>
                    </div>
                `;
            }
            reader.readAsDataURL(file);
        }
    }
    
    // Clear uploaded image
    function clearImage() {
        document.getElementById('fileInput').value = '';
        document.getElementById('imagePreview').innerHTML = '';
    }
    
    // Clear form function
    function clearForm() {
        document.getElementById('mainForm').reset();
        clearImage();
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.classList.remove('active');
    }
    
    // Add to wishlist function
    function addToWishlist(productId) {
        showLoading();
        setTimeout(() => {
            hideLoading();
            alert('Product added to wishlist!');
        }, 1000);
    }
    
    // Drag and drop functionality
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    if (uploadArea) {
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('active');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('active');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('active');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFileUpload(fileInput);
            }
        });
    }
    
    // Simulate upload progress
    function simulateUploadProgress() {
        const progressBar = document.getElementById('progressBar');
        const progressContainer = document.getElementById('uploadProgress');
        let width = 0;
        
        progressContainer.style.display = 'block';
        
        const interval = setInterval(() => {
            if (width >= 100) {
                clearInterval(interval);
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                    progressBar.style.width = '0%';
                }, 500);
            } else {
                width += 10;
                progressBar.style.width = width + '%';
            }
        }, 200);
    }
    
    // Auto-complete for keywords
    const keywordsInput = document.getElementById('keywordsInput');
    if (keywordsInput) {
        keywordsInput.addEventListener('input', function(e) {
            console.log('Searching for:', e.target.value);
        });
    }
</script>
''')
    
    return render_template_string(template)

# ==========================================
# 8. PROCESS IMAGE ROUTE
# ==========================================

@app.route('/process', methods=['POST'])
@login_required
def process():
    """Process image and find matches"""
    if 'file' not in request.files:
        flash('No file selected', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['file']
    keywords = request.form.get('keywords', '').lower().strip()
    
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('index'))
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(save_path)
        
        uploaded_image_url = f'/static/uploads/{unique_filename}'
        
        # AI Prediction
        image = Image.open(save_path)
        processed_image = prepare_image(image)
        
        if model:
            preds = model.predict(processed_image)
            decoded_preds = decode_predictions(preds, top=3)[0]
            ai_category_prediction = decoded_preds[0][1].replace('_', ' ').title()
            confidence = decoded_preds[0][2]
        else:
            ai_category_prediction = "Unknown"
            confidence = 0
        
        # Find matches
        matches = []
        for product in PRODUCT_DATABASE:
            p_cat = product['category'].lower()
            cat_match = (ai_category_prediction.lower() in p_cat) or (p_cat in ai_category_prediction.lower())
            
            key_match = False
            if keywords:
                target_str = f"{product['name']} {product['brand']} {product['category']}".lower()
                key_words = keywords.split()
                for word in key_words:
                    if word and word in target_str:
                        key_match = True
                        break
            
            if keywords:
                if key_match:
                    matches.append(product)
            else:
                if cat_match:
                    matches.append(product)
        
        # Sort by price
        if matches:
            matches.sort(key=lambda x: x['price'])
            global_best_price = matches[0]['price']
        else:
            global_best_price = None
        
        # Log search activity
        log_activity(session['user_email'], 'SEARCH', 
                    f'Found {len(matches)} products for "{ai_category_prediction}"')
        
        # Update user search history
        users = load_users()
        if session['user_email'] in users:
            if 'searches' not in users[session['user_email']]:
                users[session['user_email']]['searches'] = []
            users[session['user_email']]['searches'].append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'category': ai_category_prediction,
                'keywords': keywords,
                'matches_count': len(matches),
                'image': unique_filename
            })
            # Keep only last 50 searches
            users[session['user_email']]['searches'] = users[session['user_email']]['searches'][-50:]
            save_users(users)
        
        flash(f'Found {len(matches)} matching products', 'success')
        
        # Convert matches to string for URL
        matches_str = str(matches).replace("'", '"')  # Replace single quotes with double quotes for JSON
        
        return redirect(f'/?prediction={ai_category_prediction}&matches={matches_str}&uploaded_image_url={uploaded_image_url}&global_best_price={global_best_price}')
    
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'danger')
        log_activity(session.get('user_email', 'anonymous'), 'ERROR', f'Image processing failed: {str(e)}')
        return redirect(url_for('index'))

# ==========================================
# 9. OTHER ROUTES
# ==========================================

@app.route('/reload_csv')
@login_required
def reload_csv():
    """Reload product database"""
    global PRODUCT_DATABASE, CSV_FILES
    PRODUCT_DATABASE = load_product_database()
    CSV_FILES = [f for f in os.listdir(DATASETS_FOLDER) if f.endswith('.csv')]
    
    log_activity(session['user_email'], 'RELOAD_DB', 
                f'Loaded {len(PRODUCT_DATABASE)} products from {len(CSV_FILES)} files')
    
    flash(f'✅ Database reloaded! {len(PRODUCT_DATABASE)} products from {len(CSV_FILES)} stores', 'success')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    users = load_users()
    user_data = users.get(session['user_email'], {})
    
    # Calculate statistics
    total_searches = len(user_data.get('searches', []))
    if total_searches > 0:
        recent_searches = user_data['searches'][-5:][::-1]
        avg_matches = sum(s.get('matches_count', 0) for s in user_data['searches']) / total_searches
    else:
        recent_searches = []
        avg_matches = 0
    
    template = BASE_TEMPLATE.replace('{% block main_content %}{% endblock %}', '''
{% block main_content %}
<!-- Navbar -->
<nav class="navbar navbar-expand-lg navbar-dark fixed-top" style="background: rgba(30, 41, 59, 0.95); backdrop-filter: blur(10px);">
    <div class="container">
        <a class="navbar-brand d-flex align-items-center" href="/" style="font-weight: 800; font-size: 1.5rem;">
            <i class="bi bi-graph-up-arrow me-2" style="color: #6366f1;"></i>
            <span style="background: linear-gradient(135deg, #6366f1, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                EPC
            </span>
        </a>
        
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
            <span class="navbar-toggler-icon"></span>
        </button>
        
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav me-auto">
                <li class="nav-item">
                    <a class="nav-link" href="/">
                        <i class="bi bi-house-door me-1"></i> Home
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#features">
                        <i class="bi bi-stars me-1"></i> Features
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#upload">
                        <i class="bi bi-search me-1"></i> Compare
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link active" href="/profile">
                        <i class="bi bi-person me-1"></i> Profile
                    </a>
                </li>
            </ul>
            
            <div class="navbar-nav ms-auto">
                <div class="dropdown">
                    <a href="#" class="nav-link dropdown-toggle d-flex align-items-center" data-bs-toggle="dropdown">
                        <div class="user-avatar me-2" style="width: 32px; height: 32px; background: linear-gradient(135deg, #6366f1, #10b981); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600;">
                            ''' + (session.get('user_name', 'U')[0].upper() if session.get('user_name') else 'U') + '''
                        </div>
                        ''' + (session.get('user_name', 'User') if session.get('user_name') else 'User') + '''
                    </a>
                    <ul class="dropdown-menu dropdown-menu-end">
                        <li>
                            <a class="dropdown-item" href="/profile">
                                <i class="bi bi-person me-2"></i> Profile
                            </a>
                        </li>
                        <li>
                            <a class="dropdown-item" href="/my-searches">
                                <i class="bi bi-clock-history me-2"></i> Search History
                            </a>
                        </li>
                        <li><hr class="dropdown-divider"></li>
                        <li>
                            <a class="dropdown-item" href="/logout">
                                <i class="bi bi-box-arrow-right me-2"></i> Logout
                            </a>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
</nav>

<div class="container py-5 mt-5">
    <div class="row">
        <div class="col-lg-4 mb-4">
            <!-- User Profile Card -->
            <div class="glass-effect rounded-4 p-4 shadow">
                <div class="text-center mb-4">
                    <div class="user-avatar mx-auto mb-3" 
                         style="width: 120px; height: 120px; background: linear-gradient(135deg, #6366f1, #10b981); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 3rem; font-weight: 700;">
                        ''' + (user_data.get('full_name', 'User')[0].upper() if user_data.get('full_name') else 'U') + '''
                    </div>
                    <h4 class="mb-1">''' + (user_data.get('full_name', 'User') if user_data.get('full_name') else 'User') + '''</h4>
                    <p class="text-muted mb-2">@''' + (user_data.get('username', 'user') if user_data.get('username') else 'user') + '''</p>
                    <span class="badge ''' + ('bg-danger' if session.get('role') == 'admin' else 'bg-primary') + ''' bg-gradient p-2">
                        ''' + (session.get('role', 'user').title() if session.get('role') else 'User') + '''
                    </span>
                </div>
                
                <div class="list-group list-group-flush">
                    <div class="list-group-item d-flex justify-content-between align-items-center">
                        <span><i class="bi bi-envelope me-2"></i>Email</span>
                        <span class="text-muted">''' + (session.get('user_email', 'No email') if session.get('user_email') else 'No email') + '''</span>
                    </div>
                    <div class="list-group-item d-flex justify-content-between align-items-center">
                        <span><i class="bi bi-calendar me-2"></i>Member Since</span>
                        <span class="text-muted">''' + (user_data.get('created_at', 'N/A')[:10] if user_data.get('created_at') else 'N/A') + '''</span>
                    </div>
                    <div class="list-group-item d-flex justify-content-between align-items-center">
                        <span><i class="bi bi-clock-history me-2"></i>Last Login</span>
                        <span class="text-muted">''' + (user_data.get('last_login', 'Never')[:16] if user_data.get('last_login') else 'Never') + '''</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-lg-8">
            <!-- User Statistics -->
            <div class="row mb-4">
                <div class="col-md-4 mb-3">
                    <div class="stats-card">
                        <i class="bi bi-search display-6 mb-3"></i>
                        <span class="stats-number">''' + str(total_searches) + '''</span>
                        <p class="mb-0">Total Searches</p>
                    </div>
                </div>
                <div class="col-md-4 mb-3">
                    <div class="stats-card">
                        <i class="bi bi-graph-up display-6 mb-3"></i>
                        <span class="stats-number">''' + f"{avg_matches:.1f}" + '''</span>
                        <p class="mb-0">Avg. Matches</p>
                    </div>
                </div>
                <div class="col-md-4 mb-3">
                    <div class="stats-card">
                        <i class="bi bi-star display-6 mb-3"></i>
                        <span class="stats-number">''' + str(user_data.get('uploads_count', 0)) + '''</span>
                        <p class="mb-0">Uploads</p>
                    </div>
                </div>
            </div>
            
            <!-- Recent Activity -->
            <div class="glass-effect rounded-4 p-4 shadow mb-4">
                <h5 class="mb-4">
                    <i class="bi bi-activity me-2"></i>Recent Activity
                </h5>
                
                ''' + ('''
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Search</th>
                                <th>Results</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody>
                ''' + ''.join([f'''
                            <tr>
                                <td>{search.get("timestamp", "")[:10]}</td>
                                <td>
                                    <div class="fw-bold">{search.get("category", "")}</div>
                                    <small class="text-muted">{search.get("keywords", "No keywords")}</small>
                                </td>
                                <td>
                                    <span class="badge bg-primary">{search.get("matches_count", 0)}</span>
                                </td>
                                <td>
                                    <a href="/my-searches" class="btn btn-sm btn-outline-primary">
                                        <i class="bi bi-eye"></i>
                                    </a>
                                </td>
                            </tr>
                ''' for search in recent_searches]) + '''
                        </tbody>
                    </table>
                </div>
                ''' if recent_searches else '''
                <div class="text-center py-5">
                    <i class="bi bi-search display-1 text-muted mb-3"></i>
                    <h5>No search history yet</h5>
                    <p class="text-muted">Start searching to see your activity here</p>
                    <a href="/#upload" class="btn btn-gradient">
                        <i class="bi bi-search me-2"></i>Start Searching
                    </a>
                </div>
                ''') + '''
            </div>
        </div>
    </div>
</div>

<!-- Footer -->
<footer class="footer mt-5 py-5" style="background: rgba(30, 41, 59, 0.95); backdrop-filter: blur(10px); border-top: 1px solid rgba(255, 255, 255, 0.1);">
    <div class="container">
        <div class="row">
            <div class="col-lg-4 mb-4">
                <h5 class="text-white mb-4">
                    <i class="bi bi-graph-up-arrow me-2" style="color: #6366f1;"></i>
                    EPC System
                </h5>
                <p class="text-muted">
                    Advanced AI-powered price comparison system that helps you find the best deals across multiple e-commerce platforms.
                </p>
            </div>
            
            <div class="col-lg-2 col-6 mb-4">
                <h6 class="text-white mb-4">Quick Links</h6>
                <ul class="list-unstyled">
                    <li class="mb-2"><a href="/" class="text-muted text-decoration-none">Home</a></li>
                    <li class="mb-2"><a href="/profile" class="text-muted text-decoration-none">Profile</a></li>
                    <li class="mb-2"><a href="/my-searches" class="text-muted text-decoration-none">History</a></li>
                </ul>
            </div>
            
            <div class="col-lg-2 col-6 mb-4">
                <h6 class="text-white mb-4">Account</h6>
                <ul class="list-unstyled">
                    <li class="mb-2"><a href="/profile" class="text-muted text-decoration-none">Profile</a></li>
                    <li class="mb-2"><a href="/my-searches" class="text-muted text-decoration-none">History</a></li>
                    <li class="mb-2"><a href="/logout" class="text-muted text-decoration-none">Logout</a></li>
                </ul>
            </div>
        </div>
        
        <hr style="border-color: rgba(255, 255, 255, 0.1);">
        
        <div class="row">
            <div class="col-md-6">
                <p class="text-muted mb-0">
                    &copy; 2024 EPC System. All rights reserved.
                </p>
            </div>
        </div>
    </div>
</footer>
{% endblock %}
''')
    
    return render_template_string(template)

@app.route('/my-searches')
@login_required
def my_searches():
    """User's search history"""
    users = load_users()
    user_data = users.get(session['user_email'], {})
    searches = user_data.get('searches', [])
    
    template = BASE_TEMPLATE.replace('{% block main_content %}{% endblock %}', '''
{% block main_content %}
<!-- Navbar -->
<nav class="navbar navbar-expand-lg navbar-dark fixed-top" style="background: rgba(30, 41, 59, 0.95); backdrop-filter: blur(10px);">
    <div class="container">
        <a class="navbar-brand d-flex align-items-center" href="/" style="font-weight: 800; font-size: 1.5rem;">
            <i class="bi bi-graph-up-arrow me-2" style="color: #6366f1;"></i>
            <span style="background: linear-gradient(135deg, #6366f1, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                EPC
            </span>
        </a>
        
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
            <span class="navbar-toggler-icon"></span>
        </button>
        
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav me-auto">
                <li class="nav-item">
                    <a class="nav-link" href="/">
                        <i class="bi bi-house-door me-1"></i> Home
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#features">
                        <i class="bi bi-stars me-1"></i> Features
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#upload">
                        <i class="bi bi-search me-1"></i> Compare
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link active" href="/my-searches">
                        <i class="bi bi-clock-history me-1"></i> History
                    </a>
                </li>
            </ul>
            
            <div class="navbar-nav ms-auto">
                <div class="dropdown">
                    <a href="#" class="nav-link dropdown-toggle d-flex align-items-center" data-bs-toggle="dropdown">
                        <div class="user-avatar me-2" style="width: 32px; height: 32px; background: linear-gradient(135deg, #6366f1, #10b981); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600;">
                            ''' + (session.get('user_name', 'U')[0].upper() if session.get('user_name') else 'U') + '''
                        </div>
                        ''' + (session.get('user_name', 'User') if session.get('user_name') else 'User') + '''
                    </a>
                    <ul class="dropdown-menu dropdown-menu-end">
                        <li>
                            <a class="dropdown-item" href="/profile">
                                <i class="bi bi-person me-2"></i> Profile
                            </a>
                        </li>
                        <li>
                            <a class="dropdown-item" href="/my-searches">
                                <i class="bi bi-clock-history me-2"></i> Search History
                            </a>
                        </li>
                        <li><hr class="dropdown-divider"></li>
                        <li>
                            <a class="dropdown-item" href="/logout">
                                <i class="bi bi-box-arrow-right me-2"></i> Logout
                            </a>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
</nav>

<div class="container py-5 mt-5">
    <div class="upload-container">
        <h3 class="mb-4"><i class="bi bi-clock-history me-2"></i>My Search History</h3>
        
        ''' + ('''
        <div class="table-responsive">
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>Date & Time</th>
                        <th>Category</th>
                        <th>Keywords</th>
                        <th>Results</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
        ''' + ''.join([f'''
                    <tr>
                        <td>{search.get("timestamp", "")}</td>
                        <td>
                            <span class="badge bg-primary">{search.get("category", "")}</span>
                        </td>
                        <td>{search.get("keywords", "None")}</td>
                        <td>{search.get("matches_count", 0)} products</td>
                        <td>
                            <a href="/#upload" class="btn btn-sm btn-outline-primary">
                                <i class="bi bi-arrow-repeat"></i> Search Again
                            </a>
                        </td>
                    </tr>
        ''' for search in searches[::-1]]) + '''
                </tbody>
            </table>
        </div>
        ''' if searches else '''
        <div class="text-center py-5">
            <i class="bi bi-search display-1 text-muted mb-3"></i>
            <h5>No search history yet</h5>
            <p class="text-muted">Start searching for products to see your history here.</p>
            <a href="/#upload" class="btn btn-gradient">
                <i class="bi bi-search me-2"></i>Start Searching
            </a>
        </div>
        ''') + '''
    </div>
</div>

<!-- Footer -->
<footer class="footer mt-5 py-5" style="background: rgba(30, 41, 59, 0.95); backdrop-filter: blur(10px); border-top: 1px solid rgba(255, 255, 255, 0.1);">
    <div class="container">
        <div class="row">
            <div class="col-lg-4 mb-4">
                <h5 class="text-white mb-4">
                    <i class="bi bi-graph-up-arrow me-2" style="color: #6366f1;"></i>
                    EPC System
                </h5>
                <p class="text-muted">
                    Advanced AI-powered price comparison system that helps you find the best deals across multiple e-commerce platforms.
                </p>
            </div>
        </div>
    </div>
</footer>
{% endblock %}
''')
    
    return render_template_string(template)

# ==========================================
# 10. ADMIN ROUTES (Simplified for now)
# ==========================================

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin dashboard - simplified"""
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Admin Dashboard</title>
</head>
<body>
    <h1>Admin Dashboard</h1>
    <p>Admin panel will be implemented in the next version.</p>
    <a href="/">Back to Home</a>
</body>
</html>
''')

# ==========================================
# 11. MONITOR FUNCTION
# ==========================================

def monitor_console():
    """Display system information in console"""
    import threading
    import time

    def monitor():
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=" * 70)
            print("🛒 EPC - ECOMMERCE PRICE COMPARISON SYSTEM")
            print("=" * 70)

            users = load_users()
            logs = []

            if os.path.exists(LOGS_FILE):
                try:
                    with open(LOGS_FILE, 'r') as f:
                        logs = json.load(f)
                except:
                    pass

            print(f"\n📊 SYSTEM STATISTICS")
            print(f"   • Products in Database: {len(PRODUCT_DATABASE)}")
            print(f"   • CSV Stores: {len(CSV_FILES)}")
            print(f"   • Registered Users: {len(users)}")
            print(f"   • AI Model: {'✅ Loaded' if model else '❌ Not Available'}")

            print(f"\n📝 RECENT ACTIVITY (Last 10)")
            print("-" * 70)
            for log in logs[-10:]:
                ts = log.get('timestamp', 'N/A')[-8:]
                user = log.get('user', 'N/A')[:20]
                action = log.get('action', 'N/A')[:30]
                print(f"   {ts} | {user:20} | {action:30}")

            print("\n" + "=" * 70)
            print("⏱️  Auto-refresh every 10 seconds | Ctrl+C to stop")
            print("=" * 70)

            time.sleep(10)

    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

# ==========================================
# 12. INITIALIZATION & RUN
# ==========================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 INITIALIZING EPC SYSTEM")
    print("=" * 70)

    init_users()
    users = load_users()

    print(f"\n✅ SYSTEM READY")
    print(f"   • Admin User: admin@epc.com / admin123")
    print(f"   • Total Users: {len(users)}")
    print(f"   • Products Loaded: {len(PRODUCT_DATABASE)}")
    print(f"   • Stores Found: {len(CSV_FILES)}")
    print(f"   • AI Model: {'Loaded' if model else 'Not Available'}")

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        monitor_console()

    print("\n🌐 Server starting on http://localhost:5000")
    print("=" * 70)

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False
    )