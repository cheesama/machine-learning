from rest_framework import viewsets
from .models import Dialog
from .serializers import DialogSerializer

class DialogViewSet(viewsets.ModelViewSet):
    queryset = Dialog.objects.all()
    serializer_class = DialogSerializer
