from .models import Song

r=Song.objects.all()

for l in r:
    print(l.artist)