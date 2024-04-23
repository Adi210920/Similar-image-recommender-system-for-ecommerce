from django.shortcuts import render
from django.http import HttpResponse
from .models import Product
from django.shortcuts import render, get_object_or_404

#import for adding products
from decimal import Decimal,InvalidOperation
from django.core.management.base import BaseCommand
import csv
import sys

import logging

#train and reccomend imports
import tensorflow
from tensorflow import keras
from keras import preprocessing
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.layers import GlobalMaxPooling2D
import numpy as np
from numpy.linalg import norm
import os
import pickle
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
import cv2


model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model=tensorflow.keras.Sequential(
    [
        model,
        GlobalMaxPooling2D()
    ]

    
)

def import_products(request):
    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')

        if csv_file:
            if not csv_file.name.endswith('.csv'):
                return HttpResponse("Please upload a CSV file.")

            try:
                decoded_file = csv_file.read().decode('utf-8').splitlines()
                reader = csv.DictReader(decoded_file)

                for row in reader:
                    title = row['name']
                    image_path = row['image']

                    try:
                        actual_price = Decimal(row['actual_price'])
                    except InvalidOperation as e:
                        
                        logging.error(f"Error processing row: {e}")
                        continue

                    product = Product(title=title, price=actual_price)
                    product.image = image_path 
                    product.save()
                    logging.info(f"Product {title} saved")

                return HttpResponse("Success")

            except Exception as e:
                
                logging.error(f"Error: {e}")
                return HttpResponse("Error: " + str(e))
    return render(request,'upload_csv.html')


def home(request):
    products = Product.objects.all()
    return render(request,'home.html',{'products': products})


def get_recommendations(product):
    # Retrieve similar product IDs from the similar_products field
    similar_product_ids = product.similar_products.values_list('id', flat=True)
    
    # Retrieve the actual products from the database
    recommended_products = Product.objects.filter(id__in=similar_product_ids)
    
    return recommended_products



def product_details(request, product_id):
    # Retrieve the selected product
    product = get_object_or_404(Product, pk=product_id)

    # Get recommendations using your deep learning model
    recommended_products = get_recommendations(product)  # Pass the product object

    # Create a context dictionary
    context = {
        'product': product,
        'recommended_products': recommended_products
    }

    return render(request, 'product_details.html', context)

def extract_features(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result=result/norm(result)

    return normalized_result

def train(request):
    products = Product.objects.all()
    feature_dict = {}
    products_list = list(products)  

    
    total_iterations = len(products)

    with tqdm(total=total_iterations) as pbar:
        
        for product in products:
            feature_vector = extract_features(product.image.path, model)
            feature_dict[product.id] = feature_vector
            pbar.update(1)

    # Create a list of feature vectors
    feature_vectors = [feature_dict[product.id] for product in products]

    # Initialize the KNN model
    knn = NearestNeighbors(n_neighbors=4, metric='euclidean', n_jobs=-1)  # Find 4 nearest neighbors

    # Fit the model to the feature vectors
    knn.fit(feature_vectors)

    with tqdm(total=total_iterations) as pbar:
        for i, product in enumerate(products_list):
            # Clear existing similar products
            for similar_product in product.similar_products.all():
                product.similar_products.remove(similar_product)

            
            indices = knn.kneighbors([feature_vectors[i]], return_distance=False)[0]

        
            similar_product_ids = []
            for idx in indices[:3]:
                similar_product = products_list[idx]
                similar_product_ids.append(similar_product.id)

            product.similar_products.add(*similar_product_ids)

            pbar.update(1)

    return HttpResponse("Success")
