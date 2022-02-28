from django.shortcuts import render, redirect
import pandas as pd
from sklearn.model_selection import train_test_split
from django.contrib import messages
from .models import upload

# Create your views here.
cols=["a","b"]

df=pd.DataFrame(columns=cols)


def selectML(request):
    if "GET" == request.method:
        selectML=True
        return render(request,'index.html',{'selectMl':selectML})
    else:
        selectML=False
        ML_val = request.POST.get("mlval")
        if(ML_val == "Unsupervised Learning" or ML_val == "Reinforcement"):
            uploadfile=False
            messages.info(request,"Still Under Construction")
        else:
            uploadfile=True
        return render(request,'index.html',{'uploadfile':uploadfile,'selectMl':selectML})


def uploadfile(request):
    if "GET" == request.method:
        return render(request, 'index.html', {})
    else:
        x_axis=[]
        excel_file = request.FILES["excel_file"]
        global df
        df=pd.read_excel(excel_file)   
        problemType= True
        messages.success(request,"Successfully Uploaded file")
        messages.success(request,"Successfully Converted to DataFrame")
        return render(request, 'index.html',context={'problemType':problemType})
 
def problemType(request): 
    if "GET" == request.method:
        return render(request, 'index.html', {})
    else:
        probval=request.POST.get('probval')
        if(probval=="Regression"):
            regression=True
            columns=df.columns
            print(columns)
            return render(request,'index.html',context={'regression':regression,'columns':columns})
        elif(probval=="Classification"):
            classification=True
            return render(request,'index.html',context={'classification':classification})

        return render(request,'index.html')

def regression(request):
    x_axis=[]
    y_axis = request.POST.get("yaxis")
    testsize = request.POST.get("test_size")
    columns=df.columns
    for i in columns:
        if(i != y_axis):
            x_axis.append(i)
    X=df.drop(y_axis,axis='columns')
    y=df[y_axis]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
    messages.success(request,"Successfully Splitted into four datasets")
    return render(request,'index.html',{"yaxis":y_axis,"xaxis":x_axis,"xtrain":X_train,"xtest":X_test,"ytrain":y_train,"ytest":y_test})


def classification(request):
    return render(request,'index.html')

        # y_axis = request.POST.get("yaxis")
        # testsize = request.POST.get("test_size")
        # data=pd.read_excel(excel_file)
        # print(data.head())
        # columns=data.columns
        # for i in columns:
        #     if(i != y_axis):
        #         x_axis.append(i)
        # X=data.drop(y_axis,axis='columns')
        # y=data[y_axis]
        # X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
        # print("Done Splitting")
        #return render(request, 'index.html', {"yaxis":y_axis,"xaxis":x_axis,"xtrain":X_train,"xtest":X_test,"ytrain":y_train,"ytest":y_test})