# Similar-image-recommender-system-for-ecommerce
This Django project implements a simple product recommendation system using deep learning and k-nearest neighbors (KNN). It allows users to upload CSV files containing product data, view products, and see recommendations based on selected products

#Installation
1.Clone the repository: git clone https://github.com/your-username/your-repository.git

2.Install dependencies: pip install -r requirements.txt

3.Migrate the database: python manage.py migrate

4.Run the development server: python manage.py runserver


#Usage
Uploading Products
Visit the /import-products URL in your browser.
Upload a CSV file containing product data with columns: name, image, actual_price.
Upon successful upload, products will be saved to the database.

Viewing Products
Visit the home page to view all available products.
Click on a product to view its details and recommendations.

Training Recommendations
Access /train URL to train the recommendation system.
This process extracts features from product images and trains the KNN model.
Upon completion, the system is ready to provide recommendations.

#Dependencies
Django
TensorFlow
Keras
NumPy
scikit-learn
