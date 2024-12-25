# chatgpt_integration/admin.py
from django.contrib import admin
from .models import ConversationCategoryModel, TranslationSession

@admin.register(TranslationSession)
class TranslationSessionAdmin(admin.ModelAdmin):
    list_display = ('preferred_language','buddy_language', 'translated_text', 'suggested_response', 'timestamp')
    list_filter = ('preferred_language','buddy_language', 'timestamp')
    search_fields = ('original_text', 'translated_text', 'suggested_response')

@admin.register(ConversationCategoryModel)
class ConversationCategoryModelAdmin(admin.ModelAdmin):
    list_display = ('name','description')


