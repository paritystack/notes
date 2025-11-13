# Django

Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design. Built by experienced developers, it takes care of much of the hassle of web development, so you can focus on writing your app without needing to reinvent the wheel. It follows the "batteries-included" philosophy and provides a complete solution for web development.

## Table of Contents
- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [Project Structure](#project-structure)
- [Models and Database](#models-and-database)
- [Views](#views)
- [URL Routing](#url-routing)
- [Templates](#templates)
- [Forms](#forms)
- [Authentication](#authentication)
- [Django REST Framework](#django-rest-framework)
- [Admin Interface](#admin-interface)
- [Middleware](#middleware)
- [Static and Media Files](#static-and-media-files)
- [Testing](#testing)
- [Best Practices](#best-practices)
- [Production Deployment](#production-deployment)

---

## Introduction

**Key Features:**
- Object-Relational Mapper (ORM) for database operations
- Automatic admin interface
- Clean, pragmatic URL design
- Template engine for dynamic HTML
- Built-in authentication and authorization
- Form handling and validation
- Security features (CSRF, XSS, SQL injection protection)
- Scalable architecture
- Excellent documentation
- Large ecosystem of packages

**Use Cases:**
- Content Management Systems (CMS)
- E-commerce platforms
- Social networks
- Data-driven web applications
- RESTful APIs
- Real-time applications
- Scientific computing platforms
- Financial applications

**Philosophy:**
- Don't Repeat Yourself (DRY)
- Explicit is better than implicit
- Loose coupling and tight cohesion
- Convention over configuration

---

## Installation and Setup

### Prerequisites

```bash
# Python 3.8+ required
python3 --version
pip --version
```

### Virtual Environment Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Install Django

```bash
# Install Django
pip install django

# Verify installation
django-admin --version

# Install additional packages
pip install python-decouple psycopg2-binary pillow django-cors-headers
```

### Create New Project

```bash
# Create Django project
django-admin startproject myproject

# Navigate to project
cd myproject

# Create an app
python manage.py startapp myapp

# Run development server
python manage.py runserver

# Server runs on http://127.0.0.1:8000/
```

### Initial Database Setup

```bash
# Create initial migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
```

---

## Project Structure

```
myproject/
├── manage.py                   # Command-line utility
├── myproject/                  # Project package
│   ├── __init__.py
│   ├── settings.py            # Project settings
│   ├── urls.py                # URL declarations
│   ├── asgi.py                # ASGI entry point
│   └── wsgi.py                # WSGI entry point
├── myapp/                     # Application package
│   ├── migrations/            # Database migrations
│   ├── __init__.py
│   ├── admin.py               # Admin configuration
│   ├── apps.py                # App configuration
│   ├── models.py              # Data models
│   ├── tests.py               # Tests
│   ├── views.py               # View functions/classes
│   └── urls.py                # App URL patterns
├── templates/                 # HTML templates
├── static/                    # Static files (CSS, JS, images)
├── media/                     # User-uploaded files
└── requirements.txt           # Project dependencies
```

### Settings Configuration

**settings.py:**
```python
import os
from pathlib import Path
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config('SECRET_KEY', default='your-secret-key-here')

DEBUG = config('DEBUG', default=False, cast=bool)

ALLOWED_HOSTS = config('ALLOWED_HOSTS', default='localhost,127.0.0.1').split(',')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'myapp',  # Your app
    'rest_framework',  # For APIs
    'corsheaders',     # CORS headers
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'myproject.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('DB_NAME', default='mydb'),
        'USER': config('DB_USER', default='postgres'),
        'PASSWORD': config('DB_PASSWORD', default='password'),
        'HOST': config('DB_HOST', default='localhost'),
        'PORT': config('DB_PORT', default='5432'),
    }
}

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
```

---

## Models and Database

### Basic Model

```python
from django.db import models
from django.contrib.auth.models import User

class Category(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Categories"
        ordering = ['name']

    def __str__(self):
        return self.name

class Product(models.Model):
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('published', 'Published'),
        ('archived', 'Archived'),
    ]

    name = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='products')
    image = models.ImageField(upload_to='products/', blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    stock = models.IntegerField(default=0)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['slug']),
            models.Index(fields=['status', 'created_at']),
        ]

    def __str__(self):
        return self.name

    @property
    def is_available(self):
        return self.stock > 0 and self.status == 'published'
```

### Advanced Models

```python
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils.text import slugify

class TimestampedModel(models.Model):
    """Abstract base model with timestamp fields"""
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

class Review(TimestampedModel):
    product = models.ForeignKey('Product', on_delete=models.CASCADE, related_name='reviews')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    title = models.CharField(max_length=200)
    comment = models.TextField()
    helpful_count = models.IntegerField(default=0)

    class Meta:
        unique_together = ['product', 'user']
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.product.name} ({self.rating}★)"

class Order(TimestampedModel):
    ORDER_STATUS = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('shipped', 'Shipped'),
        ('delivered', 'Delivered'),
        ('cancelled', 'Cancelled'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='orders')
    status = models.CharField(max_length=20, choices=ORDER_STATUS, default='pending')
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    shipping_address = models.TextField()
    tracking_number = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return f"Order #{self.id} - {self.user.username}"

class OrderItem(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.PROTECT)
    quantity = models.IntegerField(validators=[MinValueValidator(1)])
    price = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return f"{self.quantity}x {self.product.name}"

    @property
    def subtotal(self):
        return self.quantity * self.price
```

### QuerySet Operations

```python
from django.db.models import Q, Count, Avg, Sum

# Basic queries
products = Product.objects.all()
product = Product.objects.get(id=1)
products = Product.objects.filter(status='published')
products = Product.objects.exclude(stock=0)

# Complex queries
products = Product.objects.filter(
    Q(name__icontains='laptop') | Q(description__icontains='laptop'),
    price__gte=500,
    status='published'
).select_related('category').prefetch_related('reviews')

# Aggregation
from django.db.models import Count, Avg
stats = Product.objects.aggregate(
    total_products=Count('id'),
    avg_price=Avg('price'),
    total_stock=Sum('stock')
)

# Annotation
categories = Category.objects.annotate(
    product_count=Count('products'),
    avg_price=Avg('products__price')
).filter(product_count__gt=0)

# Custom managers
class PublishedManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(status='published')

class Product(models.Model):
    # ... fields ...
    objects = models.Manager()
    published = PublishedManager()

# Usage
published_products = Product.published.all()
```

### Migrations

```bash
# Create migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Show migrations
python manage.py showmigrations

# Revert migration
python manage.py migrate myapp 0001

# Create empty migration
python manage.py makemigrations --empty myapp
```

---

## Views

### Function-Based Views

```python
from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.decorators import login_required
from .models import Product, Category
from .forms import ProductForm

def product_list(request):
    products = Product.objects.filter(status='published')
    categories = Category.objects.all()

    context = {
        'products': products,
        'categories': categories,
    }
    return render(request, 'products/list.html', context)

def product_detail(request, slug):
    product = get_object_or_404(Product, slug=slug, status='published')
    related_products = Product.objects.filter(
        category=product.category,
        status='published'
    ).exclude(id=product.id)[:4]

    context = {
        'product': product,
        'related_products': related_products,
    }
    return render(request, 'products/detail.html', context)

@login_required
def product_create(request):
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES)
        if form.is_valid():
            product = form.save(commit=False)
            product.created_by = request.user
            product.save()
            return redirect('product_detail', slug=product.slug)
    else:
        form = ProductForm()

    return render(request, 'products/form.html', {'form': form})

def api_products(request):
    products = Product.objects.filter(status='published').values(
        'id', 'name', 'price', 'slug'
    )
    return JsonResponse(list(products), safe=False)
```

### Class-Based Views

```python
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.urls import reverse_lazy
from .models import Product

class ProductListView(ListView):
    model = Product
    template_name = 'products/list.html'
    context_object_name = 'products'
    paginate_by = 12

    def get_queryset(self):
        queryset = Product.objects.filter(status='published')

        # Filter by category
        category_slug = self.request.GET.get('category')
        if category_slug:
            queryset = queryset.filter(category__slug=category_slug)

        # Search
        search_query = self.request.GET.get('q')
        if search_query:
            queryset = queryset.filter(
                Q(name__icontains=search_query) |
                Q(description__icontains=search_query)
            )

        return queryset.select_related('category')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = Category.objects.all()
        return context

class ProductDetailView(DetailView):
    model = Product
    template_name = 'products/detail.html'
    context_object_name = 'product'

    def get_queryset(self):
        return Product.objects.filter(status='published').select_related('category')

class ProductCreateView(LoginRequiredMixin, CreateView):
    model = Product
    form_class = ProductForm
    template_name = 'products/form.html'
    success_url = reverse_lazy('product_list')

    def form_valid(self, form):
        form.instance.created_by = self.request.user
        return super().form_valid(form)

class ProductUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Product
    form_class = ProductForm
    template_name = 'products/form.html'

    def test_func(self):
        product = self.get_object()
        return self.request.user == product.created_by or self.request.user.is_staff

    def get_success_url(self):
        return reverse_lazy('product_detail', kwargs={'slug': self.object.slug})

class ProductDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Product
    success_url = reverse_lazy('product_list')

    def test_func(self):
        product = self.get_object()
        return self.request.user == product.created_by or self.request.user.is_staff
```

---

## URL Routing

### Project URLs

**myproject/urls.py:**
```python
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('myapp.urls')),
    path('api/', include('myapp.api.urls')),
    path('accounts/', include('django.contrib.auth.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
```

### App URLs

**myapp/urls.py:**
```python
from django.urls import path
from . import views

app_name = 'products'

urlpatterns = [
    path('', views.ProductListView.as_view(), name='list'),
    path('create/', views.ProductCreateView.as_view(), name='create'),
    path('<slug:slug>/', views.ProductDetailView.as_view(), name='detail'),
    path('<slug:slug>/edit/', views.ProductUpdateView.as_view(), name='edit'),
    path('<slug:slug>/delete/', views.ProductDeleteView.as_view(), name='delete'),

    # API endpoints
    path('api/products/', views.api_products, name='api_list'),
]
```

### URL Parameters

```python
from django.urls import path, re_path
from . import views

urlpatterns = [
    # String parameter
    path('products/<slug:slug>/', views.product_detail),

    # Integer parameter
    path('products/<int:id>/', views.product_by_id),

    # UUID parameter
    path('orders/<uuid:order_id>/', views.order_detail),

    # Regular expression
    re_path(r'^articles/(?P<year>[0-9]{4})/$', views.year_archive),
]
```

---

## Templates

### Base Template

**templates/base.html:**
```django
{% load static %}
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}My Site{% endblock %}</title>
    <link rel="stylesheet" href="{% static 'css/style.css' %}">
    {% block extra_css %}{% endblock %}
</head>
<body>
    <nav>
        <a href="{% url 'products:list' %}">Products</a>
        {% if user.is_authenticated %}
            <a href="{% url 'products:create' %}">Add Product</a>
            <span>Hello, {{ user.username }}!</span>
            <a href="{% url 'logout' %}">Logout</a>
        {% else %}
            <a href="{% url 'login' %}">Login</a>
        {% endif %}
    </nav>

    <main>
        {% if messages %}
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }}">
                    {{ message }}
                </div>
            {% endfor %}
        {% endif %}

        {% block content %}{% endblock %}
    </main>

    <footer>
        <p>&copy; 2024 My Site</p>
    </footer>

    <script src="{% static 'js/main.js' %}"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>
```

### List Template

**templates/products/list.html:**
```django
{% extends 'base.html' %}
{% load static %}

{% block title %}Products{% endblock %}

{% block content %}
<div class="products-container">
    <h1>Products</h1>

    <form method="get" class="search-form">
        <input type="text" name="q" placeholder="Search products..." value="{{ request.GET.q }}">
        <select name="category">
            <option value="">All Categories</option>
            {% for category in categories %}
                <option value="{{ category.slug }}"
                    {% if request.GET.category == category.slug %}selected{% endif %}>
                    {{ category.name }}
                </option>
            {% endfor %}
        </select>
        <button type="submit">Search</button>
    </form>

    <div class="products-grid">
        {% for product in products %}
            <div class="product-card">
                {% if product.image %}
                    <img src="{{ product.image.url }}" alt="{{ product.name }}">
                {% else %}
                    <img src="{% static 'images/placeholder.png' %}" alt="No image">
                {% endif %}

                <h3>{{ product.name }}</h3>
                <p>{{ product.description|truncatewords:20 }}</p>
                <p class="price">${{ product.price }}</p>
                <a href="{% url 'products:detail' product.slug %}" class="btn">View Details</a>
            </div>
        {% empty %}
            <p>No products found.</p>
        {% endfor %}
    </div>

    {% if is_paginated %}
        <div class="pagination">
            {% if page_obj.has_previous %}
                <a href="?page=1">&laquo; First</a>
                <a href="?page={{ page_obj.previous_page_number }}">Previous</a>
            {% endif %}

            <span class="current-page">
                Page {{ page_obj.number }} of {{ page_obj.paginator.num_pages }}
            </span>

            {% if page_obj.has_next %}
                <a href="?page={{ page_obj.next_page_number }}">Next</a>
                <a href="?page={{ page_obj.paginator.num_pages }}">Last &raquo;</a>
            {% endif %}
        </div>
    {% endif %}
</div>
{% endblock %}
```

### Custom Template Tags

**myapp/templatetags/custom_tags.py:**
```python
from django import template
from django.utils.html import format_html
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter
def currency(value):
    """Format number as currency"""
    return f"${value:,.2f}"

@register.simple_tag
def star_rating(rating):
    """Display star rating"""
    full_stars = int(rating)
    half_star = 1 if rating - full_stars >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star

    stars = '★' * full_stars + '½' * half_star + '☆' * empty_stars
    return format_html('<span class="rating">{}</span>', stars)

@register.inclusion_tag('includes/product_card.html')
def product_card(product):
    """Render product card"""
    return {'product': product}
```

---

## Forms

### Model Form

```python
from django import forms
from django.core.exceptions import ValidationError
from .models import Product, Review

class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['name', 'description', 'price', 'category', 'image', 'stock', 'status']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 4}),
            'price': forms.NumberInput(attrs={'step': '0.01'}),
        }

    def clean_price(self):
        price = self.cleaned_data.get('price')
        if price and price < 0:
            raise ValidationError('Price cannot be negative')
        return price

    def clean_name(self):
        name = self.cleaned_data.get('name')
        if Product.objects.filter(name=name).exclude(pk=self.instance.pk).exists():
            raise ValidationError('Product with this name already exists')
        return name

class ReviewForm(forms.ModelForm):
    class Meta:
        model = Review
        fields = ['rating', 'title', 'comment']
        widgets = {
            'rating': forms.RadioSelect(choices=[(i, f'{i}★') for i in range(1, 6)]),
            'comment': forms.Textarea(attrs={'rows': 4, 'placeholder': 'Share your experience...'}),
        }

class SearchForm(forms.Form):
    query = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={'placeholder': 'Search products...'})
    )
    category = forms.ModelChoiceField(
        queryset=Category.objects.all(),
        required=False,
        empty_label='All Categories'
    )
    min_price = forms.DecimalField(required=False, min_value=0)
    max_price = forms.DecimalField(required=False, min_value=0)

    def clean(self):
        cleaned_data = super().clean()
        min_price = cleaned_data.get('min_price')
        max_price = cleaned_data.get('max_price')

        if min_price and max_price and min_price > max_price:
            raise ValidationError('Minimum price cannot be greater than maximum price')

        return cleaned_data
```

### Custom Validation

```python
from django import forms
from django.core.validators import EmailValidator, RegexValidator

class ContactForm(forms.Form):
    name = forms.CharField(
        max_length=100,
        validators=[
            RegexValidator(
                regex=r'^[a-zA-Z\s]+$',
                message='Name can only contain letters and spaces'
            )
        ]
    )
    email = forms.EmailField(validators=[EmailValidator()])
    phone = forms.CharField(
        validators=[
            RegexValidator(
                regex=r'^\+?1?\d{9,15}$',
                message='Enter a valid phone number'
            )
        ]
    )
    message = forms.CharField(widget=forms.Textarea)

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if email and 'spam' in email.lower():
            raise forms.ValidationError('This email appears to be spam')
        return email

    def send_email(self):
        # Send email logic here
        pass
```

---

## Authentication

### Login and Logout

```python
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
from django.contrib import messages

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome back, {user.username}!')
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password')

    return render(request, 'registration/login.html')

def user_logout(request):
    logout(request)
    messages.info(request, 'You have been logged out')
    return redirect('login')

def user_register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Registration successful!')
            return redirect('home')
    else:
        form = UserCreationForm()

    return render(request, 'registration/register.html', {'form': form})
```

### Custom User Model

```python
from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    bio = models.TextField(blank=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True)
    birth_date = models.DateField(null=True, blank=True)
    phone = models.CharField(max_length=20, blank=True)

    def __str__(self):
        return self.username

# In settings.py
AUTH_USER_MODEL = 'myapp.CustomUser'
```

### Permissions

```python
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.auth.mixins import PermissionRequiredMixin

# Function-based view
@login_required
@permission_required('myapp.add_product', raise_exception=True)
def create_product(request):
    # View logic
    pass

# Class-based view
class ProductCreateView(LoginRequiredMixin, PermissionRequiredMixin, CreateView):
    model = Product
    permission_required = 'myapp.add_product'
    # View logic

# Custom permission
class Product(models.Model):
    # ... fields ...

    class Meta:
        permissions = [
            ("can_publish", "Can publish products"),
            ("can_feature", "Can feature products"),
        ]

# Check permission in code
if request.user.has_perm('myapp.can_publish'):
    # User has permission
    pass
```

---

## Django REST Framework

### Installation

```bash
pip install djangorestframework
```

### Configuration

```python
# settings.py
INSTALLED_APPS = [
    # ...
    'rest_framework',
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticatedOrReadOnly',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10,
}
```

### Serializers

```python
from rest_framework import serializers
from .models import Product, Category, Review

class CategorySerializer(serializers.ModelSerializer):
    product_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Category
        fields = ['id', 'name', 'slug', 'description', 'product_count']

class ProductSerializer(serializers.ModelSerializer):
    category = CategorySerializer(read_only=True)
    category_id = serializers.IntegerField(write_only=True)
    reviews_count = serializers.SerializerMethodField()
    average_rating = serializers.SerializerMethodField()

    class Meta:
        model = Product
        fields = [
            'id', 'name', 'slug', 'description', 'price',
            'category', 'category_id', 'image', 'status', 'stock',
            'reviews_count', 'average_rating', 'created_at'
        ]
        read_only_fields = ['slug', 'created_at']

    def get_reviews_count(self, obj):
        return obj.reviews.count()

    def get_average_rating(self, obj):
        reviews = obj.reviews.all()
        if reviews:
            return sum(r.rating for r in reviews) / len(reviews)
        return None

    def validate_price(self, value):
        if value < 0:
            raise serializers.ValidationError('Price cannot be negative')
        return value

class ReviewSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)

    class Meta:
        model = Review
        fields = ['id', 'user', 'rating', 'title', 'comment', 'created_at']
        read_only_fields = ['user', 'created_at']
```

### API Views

```python
from rest_framework import viewsets, filters, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAuthenticatedOrReadOnly
from django_filters.rest_framework import DjangoFilterBackend
from .models import Product, Category
from .serializers import ProductSerializer, CategorySerializer

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['category', 'status']
    search_fields = ['name', 'description']
    ordering_fields = ['price', 'created_at']
    lookup_field = 'slug'

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

    @action(detail=True, methods=['post'])
    def publish(self, request, slug=None):
        product = self.get_object()
        product.status = 'published'
        product.save()
        return Response({'status': 'product published'})

    @action(detail=False, methods=['get'])
    def featured(self, request):
        featured_products = self.queryset.filter(status='published', stock__gt=0)[:10]
        serializer = self.get_serializer(featured_products, many=True)
        return Response(serializer.data)

class CategoryViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Category.objects.annotate(product_count=Count('products'))
    serializer_class = CategorySerializer
    lookup_field = 'slug'

# Function-based API view
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def product_list_create(request):
    if request.method == 'GET':
        products = Product.objects.all()
        serializer = ProductSerializer(products, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = ProductSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(created_by=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
```

---

## Admin Interface

### Basic Admin Registration

```python
from django.contrib import admin
from .models import Product, Category, Review, Order

admin.site.register(Category)
admin.site.register(Review)
```

### Custom Admin

```python
from django.contrib import admin
from django.utils.html import format_html
from .models import Product, Order, OrderItem

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'slug', 'product_count', 'created_at']
    prepopulated_fields = {'slug': ('name',)}
    search_fields = ['name']

    def product_count(self, obj):
        return obj.products.count()
    product_count.short_description = 'Products'

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ['name', 'category', 'price', 'stock', 'status', 'image_preview', 'created_at']
    list_filter = ['status', 'category', 'created_at']
    search_fields = ['name', 'description']
    prepopulated_fields = {'slug': ('name',)}
    list_editable = ['price', 'stock', 'status']
    readonly_fields = ['created_at', 'updated_at', 'image_preview']
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'slug', 'description', 'category')
        }),
        ('Pricing and Inventory', {
            'fields': ('price', 'stock', 'status')
        }),
        ('Media', {
            'fields': ('image', 'image_preview')
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def image_preview(self, obj):
        if obj.image:
            return format_html('<img src="{}" width="100" height="100" />', obj.image.url)
        return '-'
    image_preview.short_description = 'Preview'

class OrderItemInline(admin.TabularInline):
    model = OrderItem
    extra = 0
    readonly_fields = ['subtotal']

@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'status', 'total_amount', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['user__username', 'user__email', 'tracking_number']
    inlines = [OrderItemInline]
    readonly_fields = ['created_at', 'updated_at']

    actions = ['mark_as_shipped']

    def mark_as_shipped(self, request, queryset):
        queryset.update(status='shipped')
    mark_as_shipped.short_description = 'Mark selected orders as shipped'
```

---

## Middleware

```python
import time
import logging
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(MiddlewareMixin):
    def process_request(self, request):
        request.start_time = time.time()

    def process_response(self, request, response):
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            logger.info(f'{request.method} {request.path} - {response.status_code} - {duration:.2f}s')
        return response

class CustomHeaderMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response['X-Custom-Header'] = 'My Custom Value'
        return response
```

---

## Static and Media Files

### Settings

```python
# settings.py
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# For production
STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.ManifestStaticFilesStorage'
```

### Collect Static Files

```bash
python manage.py collectstatic
```

---

## Testing

### Unit Tests

```python
from django.test import TestCase
from django.contrib.auth.models import User
from .models import Product, Category

class ProductModelTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='12345')
        self.category = Category.objects.create(name='Electronics', slug='electronics')
        self.product = Product.objects.create(
            name='Laptop',
            slug='laptop',
            description='A great laptop',
            price=999.99,
            category=self.category,
            stock=10,
            created_by=self.user
        )

    def test_product_creation(self):
        self.assertEqual(self.product.name, 'Laptop')
        self.assertEqual(self.product.price, 999.99)

    def test_product_is_available(self):
        self.product.status = 'published'
        self.assertTrue(self.product.is_available)

    def test_product_str(self):
        self.assertEqual(str(self.product), 'Laptop')

class ProductViewTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='12345')
        self.category = Category.objects.create(name='Electronics', slug='electronics')
        self.product = Product.objects.create(
            name='Laptop',
            slug='laptop',
            description='A great laptop',
            price=999.99,
            category=self.category,
            status='published',
            stock=10,
            created_by=self.user
        )

    def test_product_list_view(self):
        response = self.client.get('/products/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Laptop')
        self.assertTemplateUsed(response, 'products/list.html')

    def test_product_detail_view(self):
        response = self.client.get(f'/products/{self.product.slug}/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, self.product.name)

    def test_product_create_requires_login(self):
        response = self.client.get('/products/create/')
        self.assertEqual(response.status_code, 302)

    def test_product_create_authenticated(self):
        self.client.login(username='testuser', password='12345')
        response = self.client.post('/products/create/', {
            'name': 'New Product',
            'slug': 'new-product',
            'description': 'Description',
            'price': 99.99,
            'category': self.category.id,
            'stock': 5,
            'status': 'draft'
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Product.objects.filter(name='New Product').exists())
```

---

## Best Practices

### 1. Settings Organization

```python
# settings/
# ├── __init__.py
# ├── base.py
# ├── development.py
# ├── production.py
# └── testing.py

# base.py - Common settings
# development.py
from .base import *

DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

# production.py
from .base import *

DEBUG = False
ALLOWED_HOSTS = config('ALLOWED_HOSTS').split(',')
```

### 2. Use Environment Variables

```python
from decouple import config

SECRET_KEY = config('SECRET_KEY')
DEBUG = config('DEBUG', default=False, cast=bool)
DATABASE_URL = config('DATABASE_URL')
```

### 3. Query Optimization

```python
# Use select_related for foreign keys
products = Product.objects.select_related('category').all()

# Use prefetch_related for many-to-many and reverse foreign keys
products = Product.objects.prefetch_related('reviews').all()

# Only get needed fields
products = Product.objects.values('id', 'name', 'price')

# Use iterator for large querysets
for product in Product.objects.iterator():
    # Process product
    pass
```

### 4. Security

```python
# settings.py
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# Use Django's CSRF protection
# Always validate and sanitize user input
# Use parameterized queries (Django ORM does this by default)
```

---

## Production Deployment

### Requirements File

```bash
pip freeze > requirements.txt
```

**requirements.txt:**
```
Django==4.2.7
psycopg2-binary==2.9.9
python-decouple==3.8
Pillow==10.1.0
gunicorn==21.2.0
django-cors-headers==4.3.1
djangorestframework==3.14.0
```

### Gunicorn Configuration

**gunicorn.conf.py:**
```python
bind = '0.0.0.0:8000'
workers = 4
threads = 2
worker_class = 'sync'
worker_connections = 1000
timeout = 30
keepalive = 2
accesslog = '-'
errorlog = '-'
loglevel = 'info'
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["gunicorn", "--config", "gunicorn.conf.py", "myproject.wsgi:application"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  web:
    build: .
    command: gunicorn myproject.wsgi:application --bind 0.0.0.0:8000
    volumes:
      - ./:/app
      - static_volume:/app/staticfiles
      - media_volume:/app/media
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - db

  db:
    image: postgres:15-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=mydb
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password

  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - static_volume:/app/staticfiles
      - media_volume:/app/media
    ports:
      - "80:80"
    depends_on:
      - web

volumes:
  postgres_data:
  static_volume:
  media_volume:
```

### Nginx Configuration

**nginx.conf:**
```nginx
upstream django {
    server web:8000;
}

server {
    listen 80;
    server_name example.com;

    location / {
        proxy_pass http://django;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /app/staticfiles/;
    }

    location /media/ {
        alias /app/media/;
    }
}
```

---

## Resources

**Official Documentation:**
- [Django Documentation](https://docs.djangoproject.com/)
- [Django Tutorial](https://docs.djangoproject.com/en/stable/intro/tutorial01/)
- [Django REST Framework](https://www.django-rest-framework.org/)

**Learning Resources:**
- [Django for Beginners](https://djangoforbeginners.com/)
- [Two Scoops of Django](https://www.feldroy.com/books/two-scoops-of-django-3-x)
- [Real Python Django Tutorials](https://realpython.com/tutorials/django/)

**Community:**
- [Django Forum](https://forum.djangoproject.com/)
- [Django Discord](https://discord.gg/xcRH6mN4fa)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/django)

**Tools and Packages:**
- [Django Packages](https://djangopackages.org/)
- [Awesome Django](https://github.com/wsvincent/awesome-django)
- [Django Debug Toolbar](https://django-debug-toolbar.readthedocs.io/)
