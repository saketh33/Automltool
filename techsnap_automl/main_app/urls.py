from django.urls import path
from . import views

urlpatterns = [
    path('', views.selectML, name='selectMl'),
    path('uploadfile/',views.uploadfile,name='uploadfile'),
    path('problemtype/',views.problemType, name="problemType"),
    path('typeInput/',views.typeInputs, name="typeInputs"),
    path('xcolumns/',views.xcolumns,name="xcolumns"),
    path('ycolumns/',views.ycolumns,name="ycolumns"),
    path('xtest/',views.xtest,name="xtest"),
    path('xtrain/',views.xtrain,name="xtrain"),
    path('ytest/',views.ytest,name="ytest"),
    path('ytrain/',views.ytrain,name="ytrain"),    
    path('skip/',views.skip,name="skip"),    
    path('trainalgo1/',views.algoTrain1,name="algoTrain1"),
    path('trainalgo2/',views.algoTrain2,name="algoTrain2"),
    path('checkmetrics1/',views.checkMetrics1,name="checkMetrics1"),
    path('checkmetrics2/',views.checkMetrics2,name="checkMetrics2"),
]