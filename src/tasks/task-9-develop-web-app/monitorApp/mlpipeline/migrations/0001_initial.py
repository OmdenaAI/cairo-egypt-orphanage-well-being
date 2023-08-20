# Generated by Django 4.2.4 on 2023-08-20 06:35

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('userProfile', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Activity',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('activity', models.CharField(max_length=128, verbose_name='Activity Name')),
                ('created_date', models.DateTimeField(auto_now_add=True)),
                ('updated_date', models.DateTimeField(auto_now=True)),
                ('created_by', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='activities_created_by', to=settings.AUTH_USER_MODEL)),
                ('updated_by', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='activities_updated_by', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name_plural': 'Activities',
            },
        ),
        migrations.CreateModel(
            name='Camera',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('camera_number', models.CharField(max_length=128, verbose_name='Camera Number')),
                ('room_details', models.CharField(max_length=128, verbose_name='Room Details')),
                ('connected', models.BooleanField(default=False, verbose_name='Connected')),
                ('created_date', models.DateTimeField(auto_now_add=True)),
                ('updated_date', models.DateTimeField(auto_now=True)),
                ('created_by', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='cameras_created_by', to=settings.AUTH_USER_MODEL)),
                ('updated_by', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='cameras_updated_by', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name_plural': 'Cameras',
            },
        ),
        migrations.CreateModel(
            name='Mood',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('mood', models.CharField(max_length=128, verbose_name='Mood Name')),
                ('created_date', models.DateTimeField(auto_now_add=True)),
                ('updated_date', models.DateTimeField(auto_now=True)),
                ('created_by', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='moods_created_by', to=settings.AUTH_USER_MODEL)),
                ('updated_by', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='moods_updated_by', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name_plural': 'Moods',
            },
        ),
        migrations.CreateModel(
            name='Detection',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('profile_name', models.CharField(max_length=128, verbose_name='Profile Name')),
                ('profile_role', models.CharField(max_length=128, verbose_name='Profile Role')),
                ('mood_name', models.CharField(max_length=128, verbose_name='Profile Name')),
                ('activity_name', models.CharField(max_length=128, verbose_name='Profile Name')),
                ('reference_video', models.CharField(blank=True, max_length=128, null=True, verbose_name='Reference Video')),
                ('recorded_date', models.DateTimeField()),
                ('activity', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mlpipeline.activity')),
                ('camera', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mlpipeline.camera', verbose_name='Camera')),
                ('mood', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mlpipeline.mood')),
                ('profile', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='userProfile.profile', verbose_name='Profile')),
            ],
            options={
                'verbose_name_plural': 'Detections',
            },
        ),
    ]
