from django.shortcuts import render, redirect
import pandas as pd
from sklearn.model_selection import train_test_split
from django.contrib import messages
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge


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
        showbuttons=True
        return render(request,'index.html',{"xtest":X_test.size,'showbuttons':showbuttons})


def ytest(request):
    if 'df' in request.session and 'y-axis' in request.session:
        data=request.session['df']
        y_axis=request.session['y-axis']
        testsize=request.session['test_size']
        df = pd.read_json(data)
        X=df.drop(y_axis,axis='columns')
        y=df[y_axis]
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
        showbuttons=True
        return render(request,'index.html',{"ytest":y_test.size,'showbuttons':showbuttons})

def xtrain(request):
    if 'df' in request.session and 'y-axis' in request.session:
        data=request.session['df']
        y_axis=request.session['y-axis']
        testsize=request.session['test_size']
        df = pd.read_json(data)
        X=df.drop(y_axis,axis='columns')
        y=df[y_axis]
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
        showbuttons=True
        return render(request,'index.html',{"xtrain":X_train.size,'showbuttons':showbuttons})


def ytrain(request):
    if 'df' in request.session and 'y-axis' in request.session:
        data=request.session['df']
        y_axis=request.session['y-axis']
        testsize=request.session['test_size']
        df = pd.read_json(data)
        X=df.drop(y_axis,axis='columns')
        y=df[y_axis]
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
        showbuttons=True
        return render(request,'index.html',{"ytrain":y_train.size,'showbuttons':showbuttons})

def skip(request):
    skip=True
    return render(request,'index.html',{'skip':skip})

def algoTrain(request):
    predictval=None
    skip=True
    if 'train' in request.POST:
        data=request.session['df']
        y_axis=request.session['y-axis']
        testsize=request.session['test_size']
        df = pd.read_json(data)
        X=df.drop(y_axis,axis='columns')
        y=df[y_axis]
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
        algo=request.POST.get('algoval')
        messages.info(request,"Algorithm is Getting Trained")
        if algo=="linear":
            linear=LinearRegression()
            linear.fit(X_train,y_train)
            pred=linear.predict(X_test)
        elif algo=="random":
            random=RandomForestRegressor()
            random.fit(X_train,y_train)
            pred=random.predict(X_test)
        elif algo=="decision":
            decision=DecisionTreeRegressor()
            decision.fit(X_train,y_train)
            pred=decision.predict(X_test)
        elif algo=="knn":
            knn=KNeighborsRegressor()
            knn.fit(X_train,y_train)
            pred=knn.predict(X_test)
        elif algo=="svm":
            svm=SVR()
            svm.fit(X_train,y_train)
            pred=svm.predict(X_test)
        elif algo=="ridge":
            ridge=Ridge()
            ridge.fit(X_train,y_train)
            pred=ridge.predict(X_test)    
        elif algo=="logistic":
            logistic=LogisticRegression()
            logistic.fit(X_train,y_train)
            pred=logistic.predict(X_test)  
        messages.success(request,"Successfully Trained!")
        return render(request,'index.html',{'skip':skip,'predictval':predictval})

    if "predict" in request.POST:
        checkMetrics=True
        skip=False
        data=request.session['df']
        y_axis=request.session['y-axis']
        testsize=request.session['test_size']
        df = pd.read_json(data)
        X=df.drop(y_axis,axis='columns')
        y=df[y_axis]
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
        algo=request.POST.get('algoval')
        request.session['algo']=algo
        if algo=="linear":
            linear=LinearRegression()
            linear.fit(X_train,y_train)
            pred=linear.predict(X_test)
        elif algo=="random":
            random=RandomForestRegressor()
            random.fit(X_train,y_train)
            pred=random.predict(X_test)
        elif algo=="decision":
            decision=DecisionTreeRegressor()
            decision.fit(X_train,y_train)
            pred=decision.predict(X_test)
        elif algo=="knn":
            knn=KNeighborsRegressor()
            knn.fit(X_train,y_train)
            pred=knn.predict(X_test)
        elif algo=="svm":
            svm=SVR()
            svm.fit(X_train,y_train)
            pred=svm.predict(X_test)
        elif algo=="ridge":
            ridge=Ridge()
            ridge.fit(X_train,y_train)
            pred=ridge.predict(X_test)    
        elif algo=="logistic":
            logistic=LogisticRegression()
            logistic.fit(X_train,y_train)
            pred=logistic.predict(X_test) 
        predictval=pred
        messages.success(request,"Prediction Completed!")
        return render(request,'index.html',{'skip':skip,'predictval':predictval,'checkMetrics':checkMetrics})


def checkMetrics(request):
    checkMetrics=True
    data=request.session['df']
    y_axis=request.session['y-axis']
    testsize=request.session['test_size']
    df = pd.read_json(data)
    X=df.drop(y_axis,axis='columns')
    y=df[y_axis]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
    algo=request.session['algo']
    if algo=="linear":
        linear=LinearRegression()
        linear.fit(X_train,y_train)
        pred=linear.predict(X_test)
    elif algo=="random":
        random=RandomForestRegressor()
        random.fit(X_train,y_train)
        pred=random.predict(X_test)
    elif algo=="decision":
        decision=DecisionTreeRegressor()
        decision.fit(X_train,y_train)
        pred=decision.predict(X_test)
    elif algo=="knn":
        knn=KNeighborsRegressor()
        knn.fit(X_train,y_train)
        pred=knn.predict(X_test)
    elif algo=="svm":
        svm=SVR()
        svm.fit(X_train,y_train)
        pred=svm.predict(X_test)
    elif algo=="ridge":
        ridge=Ridge()
        ridge.fit(X_train,y_train)
        pred=ridge.predict(X_test)    
    elif algo=="logistic":
        logistic=LogisticRegression()
        logistic.fit(X_train,y_train)
        pred=logistic.predict(X_test) 
    predictval=pred
    
    if "rmse" in request.POST:
        rmse=mean_squared_error(y_test, predictval, squared=False)
        return render(request,'index.html',{'checkMetrics':checkMetrics,'rmse':rmse})
 
    if "mse" in request.POST:
        mse=mean_squared_error(y_test, predictval)
        return render(request,'index.html',{'checkMetrics':checkMetrics,'mse':mse})

    if "mae" in request.POST:
        mae=mean_absolute_error(y_test, predictval)
        return render(request,'index.html',{'checkMetrics':checkMetrics,'mae':mae})

    if "r2score" in request.POST:
        r2score=r2_score(y_test, predictval)
        return render(request,'index.html',{'checkMetrics':checkMetrics,'r2score':r2score})