# Generated by Django 4.1.7 on 2023-06-23 08:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0003_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='History',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('usern', models.TextField()),
                ('s_title', models.TextField()),
                ('s_artist', models.TextField()),
            ],
        ),
    ]
