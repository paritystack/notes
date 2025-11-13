# Flask

Flask is a lightweight WSGI web application framework for Python. It's designed to make getting started quick and easy, with the ability to scale up to complex applications. Flask is often called a "microframework" because it doesn't require particular tools or libraries, giving developers flexibility in choosing their tools and architecture.

## Table of Contents
- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [Basic Application](#basic-application)
- [Routing](#routing)
- [Request and Response](#request-and-response)
- [Templates with Jinja2](#templates-with-jinja2)
- [Forms and Validation](#forms-and-validation)
- [Database Integration](#database-integration)
- [Authentication](#authentication)
- [RESTful APIs](#restful-apis)
- [Blueprints](#blueprints)
- [Error Handling](#error-handling)
- [File Uploads](#file-uploads)
- [Testing](#testing)
- [Best Practices](#best-practices)
- [Production Deployment](#production-deployment)

---

## Introduction

**Key Features:**
- Minimal core with extensions for added functionality
- Built-in development server and debugger
- Integrated unit testing support
- RESTful request dispatching
- Jinja2 templating engine
- Secure cookies for client-side sessions
- WSGI 1.0 compliant
- Unicode-based
- Extensive documentation
- Active community

**Use Cases:**
- RESTful APIs
- Microservices
- Prototypes and MVPs
- Small to medium web applications
- Backend for single-page applications
- Data science dashboards
- Webhook handlers
- Static sites with dynamic content

**Philosophy:**
- Simplicity and flexibility
- Explicit over implicit
- Start small, scale when needed
- No forced dependencies
- Easy to extend

---

## Installation and Setup

### Prerequisites

```bash
# Python 3.7+ required
python3 --version
pip --version
```

### Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Install Flask

```bash
# Install Flask
pip install Flask

# Install common extensions
pip install Flask-SQLAlchemy Flask-Migrate Flask-Login Flask-WTF
pip install Flask-CORS Flask-JWT-Extended python-dotenv
```

### Project Structure

```
flask-app/
├── app/
│   ├── __init__.py
│   ├── models.py
│   ├── routes.py
│   ├── forms.py
│   ├── templates/
│   │   ├── base.html
│   │   └── index.html
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── blueprints/
│       ├── auth/
│       └── api/
├── tests/
│   └── test_routes.py
├── migrations/
├── config.py
├── .env
├── .flaskenv
├── requirements.txt
└── run.py
```

---

## Basic Application

### Minimal Flask App

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
```

### Application Factory Pattern

**app/__init__.py:**
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from config import Config

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    # Register blueprints
    from app.blueprints.auth import auth_bp
    from app.blueprints.main import main_bp
    from app.blueprints.api import api_bp

    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')

    return app

from app import models
```

**config.py:**
```python
import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File upload settings
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
```

**run.py:**
```python
from app import create_app, db
from app.models import User, Post

app = create_app()

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'Post': Post}

if __name__ == '__main__':
    app.run(debug=True)
```

---

## Routing

### Basic Routes

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'Home Page'

@app.route('/about')
def about():
    return 'About Page'

# Route with variable
@app.route('/user/<username>')
def show_user(username):
    return f'User: {username}'

# Route with type converter
@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f'Post ID: {post_id}'

# Route with multiple types
@app.route('/path/<path:subpath>')
def show_subpath(subpath):
    return f'Subpath: {subpath}'
```

### HTTP Methods

```python
from flask import request, jsonify

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # Process login
        return {'message': 'Login successful'}
    return 'Login form'

# Separate methods
@app.get('/users')
def get_users():
    return jsonify([])

@app.post('/users')
def create_user():
    data = request.get_json()
    return jsonify(data), 201

@app.put('/users/<int:id>')
def update_user(id):
    data = request.get_json()
    return jsonify(data)

@app.delete('/users/<int:id>')
def delete_user(id):
    return '', 204
```

### URL Building

```python
from flask import url_for, redirect

@app.route('/admin')
def admin():
    return 'Admin Page'

@app.route('/redirect-to-admin')
def redirect_to_admin():
    return redirect(url_for('admin'))

@app.route('/user/<username>')
def profile(username):
    return f'Profile: {username}'

# Generate URL
with app.test_request_context():
    print(url_for('admin'))  # /admin
    print(url_for('profile', username='john'))  # /user/john
    print(url_for('static', filename='style.css'))  # /static/style.css
```

---

## Request and Response

### Request Object

```python
from flask import request, jsonify

@app.route('/search')
def search():
    # Query parameters
    query = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)

    return f'Searching for: {query}, Page: {page}'

@app.route('/submit', methods=['POST'])
def submit():
    # Form data
    name = request.form.get('name')
    email = request.form.get('email')

    # JSON data
    if request.is_json:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')

    # Files
    if 'file' in request.files:
        file = request.files['file']
        if file.filename:
            file.save(f'uploads/{file.filename}')

    # Headers
    user_agent = request.headers.get('User-Agent')
    auth_token = request.headers.get('Authorization')

    # Cookies
    session_id = request.cookies.get('session_id')

    return jsonify({
        'name': name,
        'email': email,
        'user_agent': user_agent
    })
```

### Response Object

```python
from flask import make_response, jsonify, render_template, send_file

@app.route('/json')
def json_response():
    return jsonify({
        'status': 'success',
        'data': {'id': 1, 'name': 'John'}
    })

@app.route('/custom')
def custom_response():
    response = make_response('Custom response', 200)
    response.headers['X-Custom-Header'] = 'Value'
    response.set_cookie('user_id', '123', max_age=3600)
    return response

@app.route('/download')
def download():
    return send_file('path/to/file.pdf', as_attachment=True)

@app.route('/stream')
def stream():
    def generate():
        for i in range(10):
            yield f'data: {i}\n\n'

    return app.response_class(generate(), mimetype='text/event-stream')
```

---

## Templates with Jinja2

### Base Template

**templates/base.html:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}My App{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    {% block extra_css %}{% endblock %}
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a>
        {% if current_user.is_authenticated %}
            <a href="{{ url_for('profile') }}">Profile</a>
            <a href="{{ url_for('logout') }}">Logout</a>
        {% else %}
            <a href="{{ url_for('login') }}">Login</a>
            <a href="{{ url_for('register') }}">Register</a>
        {% endif %}
    </nav>

    <main>
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        {% block content %}{% endblock %}
    </main>

    <footer>
        <p>&copy; 2024 My App</p>
    </footer>

    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>
```

### Child Template

**templates/index.html:**
```html
{% extends 'base.html' %}

{% block title %}Home - {{ super() }}{% endblock %}

{% block content %}
<h1>Welcome to {{ app_name }}</h1>

{% if users %}
    <ul>
    {% for user in users %}
        <li>
            <a href="{{ url_for('show_user', username=user.username) }}">
                {{ user.username }}
            </a>
            {% if user.is_admin %}
                <span class="badge">Admin</span>
            {% endif %}
        </li>
    {% endfor %}
    </ul>
{% else %}
    <p>No users found.</p>
{% endif %}

<!-- Macros -->
{% macro render_user(user) %}
    <div class="user-card">
        <h3>{{ user.username }}</h3>
        <p>{{ user.email }}</p>
    </div>
{% endmacro %}

{% for user in users %}
    {{ render_user(user) }}
{% endfor %}
{% endblock %}
```

### Template Filters and Functions

```python
from flask import Flask
from datetime import datetime

app = Flask(__name__)

@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d %H:%M'):
    return value.strftime(format)

@app.template_filter('currency')
def currency(value):
    return f'${value:,.2f}'

@app.context_processor
def utility_processor():
    def format_price(amount):
        return f'${amount:,.2f}'

    return dict(format_price=format_price)

# Usage in template:
# {{ order.created_at|datetimeformat }}
# {{ product.price|currency }}
# {{ format_price(100.50) }}
```

---

## Forms and Validation

### Flask-WTF Forms

```python
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, TextAreaField, SelectField, BooleanField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
from app.models import User

class RegistrationForm(FlaskForm):
    username = StringField('Username',
                         validators=[DataRequired(), Length(min=3, max=20)])
    email = StringField('Email',
                       validators=[DataRequired(), Email()])
    password = PasswordField('Password',
                           validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password',
                                   validators=[DataRequired(), EqualTo('password')])

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username already exists')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email already registered')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')

class PostForm(FlaskForm):
    title = StringField('Title', validators=[DataRequired(), Length(max=100)])
    content = TextAreaField('Content', validators=[DataRequired()])
    category = SelectField('Category', coerce=int)

    def __init__(self, *args, **kwargs):
        super(PostForm, self).__init__(*args, **kwargs)
        from app.models import Category
        self.category.choices = [(c.id, c.name) for c in Category.query.all()]
```

### Form Handling in Views

```python
from flask import render_template, redirect, url_for, flash
from app import db
from app.forms import RegistrationForm, LoginForm
from app.models import User

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()

    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()

        flash('Registration successful!', 'success')
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()

        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            flash('Login successful!', 'success')

            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))

        flash('Invalid email or password', 'danger')

    return render_template('login.html', form=form)
```

### Form Template

**templates/register.html:**
```html
{% extends 'base.html' %}

{% block content %}
<h2>Register</h2>
<form method="POST" novalidate>
    {{ form.hidden_tag() }}

    <div class="form-group">
        {{ form.username.label }}
        {{ form.username(class='form-control') }}
        {% if form.username.errors %}
            <div class="errors">
                {% for error in form.username.errors %}
                    <span>{{ error }}</span>
                {% endfor %}
            </div>
        {% endif %}
    </div>

    <div class="form-group">
        {{ form.email.label }}
        {{ form.email(class='form-control') }}
        {% if form.email.errors %}
            <div class="errors">
                {% for error in form.email.errors %}
                    <span>{{ error }}</span>
                {% endfor %}
            </div>
        {% endif %}
    </div>

    <div class="form-group">
        {{ form.password.label }}
        {{ form.password(class='form-control') }}
        {% if form.password.errors %}
            <div class="errors">
                {% for error in form.password.errors %}
                    <span>{{ error }}</span>
                {% endfor %}
            </div>
        {% endif %}
    </div>

    <div class="form-group">
        {{ form.confirm_password.label }}
        {{ form.confirm_password(class='form-control') }}
    </div>

    <button type="submit" class="btn btn-primary">Register</button>
</form>
{% endblock %}
```

---

## Database Integration

### SQLAlchemy Models

**app/models.py:**
```python
from app import db, login_manager
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    posts = db.relationship('Post', backref='author', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    slug = db.Column(db.String(120), unique=True, index=True)
    published = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('category.id'))

    def __repr__(self):
        return f'<Post {self.title}>'

class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    posts = db.relationship('Post', backref='category', lazy=True)
```

### Database Operations

```python
from app import db
from app.models import User, Post

# Create
user = User(username='john', email='john@example.com')
user.set_password('password123')
db.session.add(user)
db.session.commit()

# Read
users = User.query.all()
user = User.query.filter_by(username='john').first()
user = User.query.get(1)
posts = Post.query.filter_by(published=True).order_by(Post.created_at.desc()).all()

# Update
user = User.query.get(1)
user.email = 'newemail@example.com'
db.session.commit()

# Delete
user = User.query.get(1)
db.session.delete(user)
db.session.commit()

# Complex queries
from sqlalchemy import or_, and_

posts = Post.query.filter(
    or_(
        Post.title.like('%python%'),
        Post.content.like('%python%')
    ),
    Post.published == True
).all()

# Pagination
page = request.args.get('page', 1, type=int)
posts = Post.query.order_by(Post.created_at.desc()).paginate(
    page=page, per_page=10, error_out=False
)
```

### Migrations

```bash
# Initialize migrations
flask db init

# Create migration
flask db migrate -m "Add user table"

# Apply migration
flask db upgrade

# Rollback
flask db downgrade
```

---

## Authentication

### Flask-Login Setup

```python
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from app import app, db
from app.models import User

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()

        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))

        flash('Invalid email or password', 'danger')

    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)
```

### JWT Authentication

```python
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity

app.config['JWT_SECRET_KEY'] = 'your-secret-key'
jwt = JWTManager(app)

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()

    if user and user.check_password(password):
        access_token = create_access_token(identity=user.id)
        return jsonify(access_token=access_token), 200

    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/api/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    return jsonify(username=user.username), 200
```

---

## RESTful APIs

### Flask-RESTful

```python
from flask import Flask
from flask_restful import Resource, Api, reqparse, fields, marshal_with
from app import db
from app.models import Post

app = Flask(__name__)
api = Api(app)

# Request parser
post_parser = reqparse.RequestParser()
post_parser.add_argument('title', type=str, required=True, help='Title is required')
post_parser.add_argument('content', type=str, required=True)
post_parser.add_argument('category_id', type=int)

# Resource fields for serialization
post_fields = {
    'id': fields.Integer,
    'title': fields.String,
    'content': fields.String,
    'created_at': fields.DateTime(dt_format='iso8601'),
    'author': fields.Nested({
        'id': fields.Integer,
        'username': fields.String
    })
}

class PostListAPI(Resource):
    @marshal_with(post_fields)
    def get(self):
        posts = Post.query.all()
        return posts

    @marshal_with(post_fields)
    def post(self):
        args = post_parser.parse_args()
        post = Post(
            title=args['title'],
            content=args['content'],
            user_id=current_user.id,
            category_id=args.get('category_id')
        )
        db.session.add(post)
        db.session.commit()
        return post, 201

class PostAPI(Resource):
    @marshal_with(post_fields)
    def get(self, post_id):
        post = Post.query.get_or_404(post_id)
        return post

    @marshal_with(post_fields)
    def put(self, post_id):
        post = Post.query.get_or_404(post_id)
        args = post_parser.parse_args()
        post.title = args['title']
        post.content = args['content']
        db.session.commit()
        return post

    def delete(self, post_id):
        post = Post.query.get_or_404(post_id)
        db.session.delete(post)
        db.session.commit()
        return '', 204

api.add_resource(PostListAPI, '/api/posts')
api.add_resource(PostAPI, '/api/posts/<int:post_id>')
```

---

## Blueprints

### Creating Blueprints

**app/blueprints/auth/__init__.py:**
```python
from flask import Blueprint

auth_bp = Blueprint('auth', __name__)

from app.blueprints.auth import routes
```

**app/blueprints/auth/routes.py:**
```python
from flask import render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required
from app.blueprints.auth import auth_bp
from app import db
from app.models import User
from app.forms import LoginForm, RegistrationForm

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            return redirect(url_for('main.index'))
        flash('Invalid credentials', 'danger')
    return render_template('auth/login.html', form=form)

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful!', 'success')
        return redirect(url_for('auth.login'))
    return render_template('auth/register.html', form=form)
```

### Registering Blueprints

**app/__init__.py:**
```python
def create_app():
    app = Flask(__name__)

    # Register blueprints
    from app.blueprints.auth import auth_bp
    from app.blueprints.main import main_bp
    from app.blueprints.api import api_bp

    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')

    return app
```

---

## Error Handling

```python
from flask import render_template, jsonify

@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('errors/500.html'), 500

@app.errorhandler(403)
def forbidden_error(error):
    return jsonify({'error': 'Forbidden'}), 403

# Custom exception
class ValidationError(Exception):
    pass

@app.errorhandler(ValidationError)
def handle_validation_error(error):
    return jsonify({'error': str(error)}), 400
```

---

## File Uploads

```python
import os
from werkzeug.utils import secure_filename
from flask import request, flash, redirect, url_for

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            flash('File uploaded successfully', 'success')
            return redirect(url_for('index'))

    return render_template('upload.html')
```

---

## Testing

```python
import unittest
from app import create_app, db
from app.models import User, Post
from config import TestingConfig

class UserModelTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app(TestingConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_password_hashing(self):
        user = User(username='john', email='john@example.com')
        user.set_password('password')
        self.assertFalse(user.check_password('wrong'))
        self.assertTrue(user.check_password('password'))

class RoutesTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app(TestingConfig)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_index_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_login(self):
        # Create user
        user = User(username='test', email='test@example.com')
        user.set_password('password')
        db.session.add(user)
        db.session.commit()

        # Test login
        response = self.client.post('/auth/login', data={
            'email': 'test@example.com',
            'password': 'password'
        }, follow_redirects=True)

        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
```

---

## Best Practices

### 1. Application Factory

```python
def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)

    return app
```

### 2. Configuration Management

```python
# Use environment variables
from dotenv import load_dotenv
load_dotenv()

SECRET_KEY = os.environ.get('SECRET_KEY')
DATABASE_URL = os.environ.get('DATABASE_URL')
```

### 3. Error Handling

```python
# Always handle exceptions
try:
    # Database operation
    db.session.commit()
except Exception as e:
    db.session.rollback()
    app.logger.error(f'Error: {str(e)}')
    flash('An error occurred', 'danger')
```

### 4. Security

```python
# CSRF protection
from flask_wtf.csrf import CSRFProtect
csrf = CSRFProtect(app)

# Security headers
from flask_talisman import Talisman
Talisman(app, content_security_policy=None)

# Rate limiting
from flask_limiter import Limiter
limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/api/data')
@limiter.limit("5 per minute")
def api_data():
    return jsonify({'data': []})
```

---

## Production Deployment

### Requirements

**requirements.txt:**
```
Flask==3.0.0
Flask-SQLAlchemy==3.1.1
Flask-Migrate==4.0.5
Flask-Login==0.6.3
Flask-WTF==1.2.1
Flask-CORS==4.0.0
Flask-JWT-Extended==4.5.3
python-dotenv==1.0.0
gunicorn==21.2.0
psycopg2-binary==2.9.9
```

### Gunicorn

```bash
# Install
pip install gunicorn

# Run
gunicorn -w 4 -b 0.0.0.0:8000 "app:create_app()"

# With config file
gunicorn -c gunicorn.conf.py "app:create_app()"
```

**gunicorn.conf.py:**
```python
bind = '0.0.0.0:8000'
workers = 4
threads = 2
timeout = 30
accesslog = '-'
errorlog = '-'
loglevel = 'info'
```

### Docker

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:create_app()"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
    depends_on:
      - db

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

## Resources

**Official Documentation:**
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Flask Mega-Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)
- [Flask Extensions](https://flask.palletsprojects.com/en/latest/extensions/)

**Extensions:**
- [Flask-SQLAlchemy](https://flask-sqlalchemy.palletsprojects.com/)
- [Flask-Login](https://flask-login.readthedocs.io/)
- [Flask-WTF](https://flask-wtf.readthedocs.io/)
- [Flask-RESTful](https://flask-restful.readthedocs.io/)

**Community:**
- [Flask Discord](https://discord.gg/pallets)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/flask)
- [Reddit r/flask](https://www.reddit.com/r/flask/)

**Books:**
- Flask Web Development by Miguel Grinberg
- Flask Framework Cookbook
