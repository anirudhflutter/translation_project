# Generated by Django 4.2.16 on 2024-12-10 08:34

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('translator', '0002_conversationcategorymodel'),
    ]

    operations = [
        migrations.AddField(
            model_name='translationsession',
            name='category',
            field=models.ForeignKey(default='', on_delete=django.db.models.deletion.CASCADE, related_name='sessions', to='translator.conversationcategorymodel'),
        ),
    ]