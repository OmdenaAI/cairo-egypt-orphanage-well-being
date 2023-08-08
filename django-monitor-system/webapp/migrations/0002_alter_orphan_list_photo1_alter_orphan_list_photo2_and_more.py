# Generated by Django 4.2.4 on 2023-08-08 13:12

from django.db import migrations, models
import webapp.models


class Migration(migrations.Migration):

    dependencies = [
        ('webapp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='orphan_list',
            name='photo1',
            field=models.ImageField(max_length=300, upload_to=webapp.models.Rename.save),
        ),
        migrations.AlterField(
            model_name='orphan_list',
            name='photo2',
            field=models.ImageField(blank=True, max_length=300, upload_to=webapp.models.Rename.save),
        ),
        migrations.AlterField(
            model_name='orphan_list',
            name='photo3',
            field=models.ImageField(blank=True, max_length=300, upload_to=webapp.models.Rename.save),
        ),
    ]