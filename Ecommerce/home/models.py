# models.py

from django.db import models

class Product(models.Model):
    title = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    image = models.ImageField(upload_to='product_images/')
    similar_products = models.ManyToManyField('self', blank=True)

    def __str__(self):
        return self.title

    def delete(self, *args, **kwargs):
        if self.image:
            image_path = self.image.path
            if os.path.exists(image_path):
                os.remove(image_path)
        super(Product, self).delete(*args, **kwargs)
