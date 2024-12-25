from django.contrib import admin
from .models import QuizQuestionModel, QuizMemeModel

class QuestionAdmin(admin.ModelAdmin):
    list_display = ('text', 'get_memes')  # Add 'get_memes' method to list_display

    def get_memes(self, obj):
        # This method returns a comma-separated list of meme texts
        return ", ".join([meme.text for meme in obj.meme_choices.all()])
    get_memes.short_description = 'Memes'  # Change column header to 'Memes'
    
# Register models with admin
admin.site.register(QuizQuestionModel, QuestionAdmin)
admin.site.register(QuizMemeModel)