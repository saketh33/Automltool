from django.urls import path
from . import views

urlpatterns = [
    path('', views.selectML, name='selectMl'),
    path('uploadfile/',views.uploadfile,name='uploadfile'),
    path('problemtype/',views.problemType, name="problemType"),
    path('regression/',views.regression, name="regression"),
    path('classification/',views.classification, name="classification"),
    path('xcolumns/',views.xcolumns,name="xcolumns"),
    path('ycolumns/',views.ycolumns,name="ycolumns"),
    path('xtest/',views.xtest,name="xtest"),
    path('xtrain/',views.xtrain,name="xtrain"),
    path('ytest/',views.ytest,name="ytest"),
    path('ytrain/',views.ytrain,name="ytrain"),    
]