# yourapp/management/commands/add_categories.py
from django.core.management.base import BaseCommand
from translator.models import ConversationCategoryModel

class Command(BaseCommand):
    help = 'Add default conversation categories to the database'

    def handle(self, *args, **kwargs):
        categories_data = [
            ('German Traditions and Festivals', 'Talk about holidays, customs, and local traditions (e.g., Oktoberfest, Christmas traditions, etc.).'),
            ('Cultural Differences', 'Conversations about the contrasts between your home country and Germany, including customs, lifestyle, and behavior.'),
            ('German History', 'Discussion topics related to German history, key events, and historical figures.'),
            ('Art and Music', 'Conversations about famous German artists, musicians, and cultural contributions (e.g., Bach, Beethoven, Goethe).'),
            ('Cuisine', 'Talking about German food, regional specialties, and food etiquette.'),
            ('Weather', 'Casual talks about the weather, which is a common conversation starter in Germany.'),
            ('Transportation', 'Conversations about public transport, how to get around in the city, biking, or driving in Germany.'),
            ('Shopping and Markets', 'Discussions about local stores, open-air markets, or shopping habits in Germany.'),
            ('Daily Routines', 'What an average day in Germany looks like (e.g., breakfast habits, workday schedules, leisure time).'),
            ('Workplace Etiquette', 'Discussing job roles, company cultures, office environment, and professional norms.'),
            ('University Life', 'Conversations about academic institutions, student life, classes, and university-related experiences.'),
            ('Job Hunting and Careers', 'Talks on how to apply for jobs, internships, and the job market in Germany.'),
            ('Work-Life Balance', 'How people manage work and personal life, vacation time, and family leave.'),
            ('Healthcare System', 'Conversations about the German healthcare system, doctors, hospitals, and medical insurance.'),
            ('Fitness and Sports', 'Discussions about popular sports in Germany, like soccer, cycling, running, or gym culture.'),
            ('Mental Health', 'Conversations on mental well-being, how Germans approach mental health, therapy, and support systems.'),
            ('Making Friends', 'Conversations about meeting new people, socializing, and building friendships in Germany.'),
            ('Dating and Relationships', 'Discussions around dating culture, relationships, and social norms in Germany.'),
            ('Family Dynamics', 'Talk about family life in Germany, parenting styles, and generational differences.'),
            ('Gender Roles and Equality', 'Topics on gender equality, feminism, and social movements in Germany.'),
            ('Current Affairs', 'Conversations about the latest political developments in Germany and around the world.'),
            ('Immigration and Integration', 'Topics about integrating into German society as an immigrant and the challenges faced by newcomers.'),
            ('German Government and Policies', 'Discussions about German politics, policies, and how the political system works.'),
            ('Sustainability and Environment', 'Conversations on environmental policies, renewable energy, and sustainability practices in Germany.'),
            ('Tech Industry', 'Conversations about the technology and innovation scene in Germany (e.g., Berlin’s startup culture, Germany’s role in tech development).'),
            ('Smart Cities and Urban Living', 'Discussions about the future of cities, tech in daily life, and smart homes.'),
            ('Social Media Trends', 'How Germans use social media, the difference in social media behavior, and online culture in Germany.'),
            ('Tourist Spots in Germany', 'Conversations about popular travel destinations and hidden gems in Germany.'),
            ('Outdoor Activities', 'Hiking, cycling, and nature walks that are common in Germany, and outdoor adventure conversations.'),
            ('Travel in Europe', 'Talking about traveling within Europe, train journeys, and exploring nearby countries.'),
            ('Finding Housing', 'Conversations on renting apartments, dealing with landlords, and finding a place to live in Germany.')
        ]
        
        for name, description in categories_data:
            category, created = ConversationCategoryModel.objects.get_or_create(name=name, description=description)
            if created:
                self.stdout.write(self.style.SUCCESS(f'Category "{name}" added successfully!'))
            else:
                self.stdout.write(self.style.SUCCESS(f'Category "{name}" already exists.'))
