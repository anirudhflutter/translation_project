from django.shortcuts import render
from .models import QuizQuestionModel

def personality_quiz(request):
    # Fetch all questions with their associated meme choices
    questions = QuizQuestionModel.objects.all()

    # Check if the user has submitted an answer (optional, for handling form submission)
    if request.method == 'POST':
        # Process the answers (You can implement this part based on your logic)
        selected_answers = request.POST.getlist('selected_answers')  # List of selected meme IDs
        # You can save this to the database or do further processing here

    return render(request, 'quiz.html', {
        'questions': questions,
    })
