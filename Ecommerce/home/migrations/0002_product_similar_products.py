# Generated by Django 4.2 on 2023-10-19 10:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='product',
            name='similar_products',
            field=models.ManyToManyField(blank=True, to='home.product'),
        ),
    ]
