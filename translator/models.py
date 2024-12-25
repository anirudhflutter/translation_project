# chatgpt_integration/models.py
from django.db import models

class ConversationCategoryModel(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField()

    def __str__(self):
        return self.name
    
class TranslationSession(models.Model):
    LANGUAGE_CHOICES = [
        ('de_to_en', 'German to English'),
        ('en_to_de', 'English to German'),
    ]

    preferred_language = models.CharField(max_length=10, choices=LANGUAGE_CHOICES,default='')
    buddy_language = models.CharField(max_length=10, choices=LANGUAGE_CHOICES,default='')
    original_text = models.TextField()
    translated_text = models.TextField()
    suggested_response = models.TextField()
    category = models.ForeignKey(ConversationCategoryModel, on_delete=models.CASCADE, related_name='sessions',default="")  # ForeignKey to category
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.get_language_direction_display()} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

