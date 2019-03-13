from rest_framework import routers
from annotator.viewsets import DialogViewSet

router = routers.DefaultRouter()

router.register(r'dialog', DialogViewSet)
