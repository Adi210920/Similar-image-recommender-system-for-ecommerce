from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('',views.home,name='home'),
    path('product/<int:product_id>/', views.product_details, name='product_details'),
    path('train',views.train,name='train'),
    path('importproducts',views.import_products,name='importproducts')
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)