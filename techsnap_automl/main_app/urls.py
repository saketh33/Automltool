from django.urls import path
from . import views

urlpatterns = [
    path('', views.selectML, name='selectMl'),
    path('uploadfile/',views.uploadfile,name='uploadfile'),
    path('problemtype/',views.problemType, name="problemType"),
    path('regression/',views.regression, name="regression"),
    path('classification/',views.classification, name="classification"),
]