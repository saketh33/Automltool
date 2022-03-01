from django.shortcuts import render, redirect
import pandas as pd
from sklearn.model_selection import train_test_split
from django.contrib import messages

def selectML(request):
    if "GET" == request.method:
        selectML=True
        return render(request,'index.html',{'selectMl':selectML})
    else:
        selectML=False
        ML_val = request.POST.get("mlval")
        if(ML_val == "Unsupervised Learning" or ML_val == "Reinforcement"):
            selectML=True
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
        upload_file = request.FILES["upload_file"]
        try:
            df = pd.read_excel(upload_file)
        except:
            df = pd.read_csv(upload_file)
        df = df.to_json()
        request.session['df']=df
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
            if 'df' in request.session:
                data=request.session['df']
                df = pd.read_json(data)
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
    if 'df' in request.session:
        data=request.session['df']
        df = pd.read_json(data)
        columns=df.columns
        for i in columns:
            if(i != y_axis):
                x_axis.append(i)
        request.session['x-axis']=x_axis
        request.session['y-axis']=y_axis
        request.session['test_size']=testsize
        X=df.drop(y_axis,axis='columns')
        y=df[y_axis]
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
        messages.success(request,"Successfully Splitted into four datasets")
        showbuttons=True
        return render(request,'index.html',{'showbuttons':showbuttons})


def classification(request):
    return render(request,'index.html')


def xcolumns(request):
    showbuttons=True
    if 'x-axis' in request.session:
        x_axis=request.session['x-axis']
        return render(request,'index.html',{"xaxis":x_axis,"showbuttons":showbuttons})


def ycolumns(request):
    showbuttons=True
    if 'y-axis' in request.session:
        y_axis=request.session['y-axis']
        return render(request,'index.html',{"yaxis":y_axis,"showbuttons":showbuttons})

def xtest(request):
    if 'df' in request.session and 'y-axis' in request.session:
        data=request.session['df']
        y_axis=request.session['y-axis']
        testsize=request.session['test_size']
        df = pd.read_json(data)
        X=df.drop(y_axis,axis='columns')
        y=df[y_axis]
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
        messages.success(request,"Successfully Splitted into four datasets")
        showbuttons=True
        return render(request,'index.html',{"xtest":X_test,'showbuttons':showbuttons})


def ytest(request):
    if 'df' in request.session and 'y-axis' in request.session:
        data=request.session['df']
        y_axis=request.session['y-axis']
        testsize=request.session['test_size']
        df = pd.read_json(data)
        X=df.drop(y_axis,axis='columns')
        y=df[y_axis]
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
        messages.success(request,"Successfully Splitted into four datasets")
        showbuttons=True
        return render(request,'index.html',{"ytest":y_test,'showbuttons':showbuttons})

def xtrain(request):
    if 'df' in request.session and 'y-axis' in request.session:
        data=request.session['df']
        y_axis=request.session['y-axis']
        testsize=request.session['test_size']
        df = pd.read_json(data)
        X=df.drop(y_axis,axis='columns')
        y=df[y_axis]
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
        messages.success(request,"Successfully Splitted into four datasets")
        showbuttons=True
        return render(request,'index.html',{"xtrain":X_train,'showbuttons':showbuttons})


def ytrain(request):
    if 'df' in request.session and 'y-axis' in request.session:
        data=request.session['df']
        y_axis=request.session['y-axis']
        testsize=request.session['test_size']
        df = pd.read_json(data)
        X=df.drop(y_axis,axis='columns')
        y=df[y_axis]
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
        messages.success(request,"Successfully Splitted into four datasets")
        showbuttons=True
        return render(request,'index.html',{"ytrain":y_train,'showbuttons':showbuttons})