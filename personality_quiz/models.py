from django.db import models

# Define the Meme model (which will hold meme image and text)
class QuizMemeModel(models.Model):
    image = models.ImageField(upload_to='memes/')  # Path to store meme images
    text = models.CharField(max_length=255)  # Text caption for the meme
    
    def __str__(self):
        return self.text  # Return the meme text as the string representation

# Define the Question model
class QuizQuestionModel(models.Model):
    text = models.CharField(max_length=500)  # The text for the question
    meme_choices = models.ManyToManyField(QuizMemeModel, related_name='questions')  
    
    def __str__(self):
        return self.text  # Return the question text as string representation

# Define the Answer model to store user answers
class QuizAnswerModel(models.Model):
    question = models.ForeignKey(QuizQuestionModel, on_delete=models.CASCADE)  # Link to the question
    meme = models.ForeignKey(QuizMemeModel, on_delete=models.CASCADE)  # Link to the meme choice selected
    user_response = models.CharField(max_length=255)  # Store response text (optional)
    
    def __str__(self):
        return f"Answer for {self.question.text} - {self.meme.text}"
