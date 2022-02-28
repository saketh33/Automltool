from django.shortcuts import render
import openpyxl
import pandas as pd
from sklearn.model_selection import train_test_split

# Create your views here.
def home(request):
    if "GET" == request.method:
        return render(request, 'index.html', {})
    else:
        x_axis=[]
        excel_file = request.FILES["excel_file","file"]
        y_axis = request.POST.get("yaxis")
        testsize = request.POST.get("test_size")
        data=pd.read_excel(excel_file)
        print(data.head())
        columns=data.columns
        for i in columns:
            if(i != y_axis):
                x_axis.append(i)
        X=data.drop(y_axis,axis='columns')
        y=data[y_axis]
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
        print("Done Splitting")

        return render(request, 'index.html', {"yaxis":y_axis,"xaxis":x_axis,"xtrain":X_train,"xtest":X_test,"ytrain":y_train,"ytest":y_test})
