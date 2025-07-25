from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('core', '0004_user_date_of_birth_user_first_name_user_gender_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='preferred_llm',
            field=models.CharField(default='openai', max_length=50),
        ),
    ]