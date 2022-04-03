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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import log_loss 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

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
            probType="Regression"
        elif(probval=="Classification"):
            probType="Classification"
        request.session['probType']=probType
        data=request.session['df']
        df = pd.read_json(data)
        columns=df.columns
        print(columns)
        return render(request,'index.html',context={'probType':probType,'columns':columns})

def typeInputs(request):
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
    probType=request.session['probType']
    if probType=="Regression":
        Regression=True
        Classification=False
    else:
        Classification=True
        Regression=False
    return render(request,'index.html',{'Regression':Regression,'Classification':Classification})

def algoTrain1(request):
    predictval=None
    if 'train' in request.POST:
        Regression=True
        messages.info(request,"Algorithm is Getting Trained")
        messages.success(request,"Successfully Trained!")
        return render(request,'index.html',{'Regression':Regression,'predictval':predictval})

    if "predict" in request.POST:
        checkMetrics1=True
        Regression=False
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
        messages.success(request,"Prediction Completed!")
        return render(request,'index.html',{'checkMetrics1':checkMetrics1,'Regression':Regression,'predictval':pred})


def algoTrain2(request):
    predictval=None
    if 'train' in request.POST:
        Classification=True
        messages.info(request,"Algorithm is Getting Trained")
        messages.success(request,"Successfully Trained!")
        return render(request,'index.html',{'Classification':Classification,'predictval':predictval})

    if "predict" in request.POST:
        checkMetrics2=True
        Classification=False
        data=request.session['df']
        y_axis=request.session['y-axis']
        testsize=request.session['test_size']
        df = pd.read_json(data)
        X=df.drop(y_axis,axis='columns')
        y=df[y_axis]
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
        algo=request.POST.get('algoval')
        request.session['algo']=algo
        if algo=="AdaBoostClassifier":
            ada=AdaBoostClassifier()
            ada.fit(X_train,y_train)
            pred=ada.predict(X_test)
        elif algo=="GradientBoostingClassifier":
            grad=GradientBoostingClassifier()
            grad.fit(X_train,y_train)
            pred=grad.predict(X_test)
        elif algo=="BaggingClassifier":
            bag=BaggingClassifier()
            bag.fit(X_train,y_train)
            pred=bag.predict(X_test)
        elif algo=="ExtraTreesClassifier":
            extraTree=ExtraTreesClassifier()
            extraTree.fit(X_train,y_train)
            pred=extraTree.predict(X_test)
        elif algo=="DecisionTreeClassifier":
            dec=DecisionTreeClassifier()
            dec.fit(X_train,y_train)
            pred=dec.predict(X_test)
        elif algo=="RadiusNeighborsClassifier":
            radius=RadiusNeighborsClassifier()
            radius.fit(X_train,y_train)
            pred=radius.predict(X_test)    
        elif algo=="KNeighborsClassifier":
            kneigh=KNeighborsClassifier()
            kneigh.fit(X_train,y_train)
            pred=kneigh.predict(X_test) 
        elif algo=="Logistics":
            logi=LogisticRegression()
            logi.fit(X_train,y_train)
            pred=logi.predict(X_test) 
         
        messages.success(request,"Prediction Completed!")
        return render(request,'index.html',{'checkMetrics2':checkMetrics2,'Classification':Classification,'predictval':pred})


def checkMetrics1(request):
    checkMetrics1=True
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
    predictval=pred
    
    if "rmse" in request.POST:
        rmse=mean_squared_error(y_test, predictval, squared=False)
        return render(request,'index.html',{'checkMetrics1':checkMetrics1,'rmse':rmse})
 
    if "mse" in request.POST:
        mse=mean_squared_error(y_test, predictval)
        return render(request,'index.html',{'checkMetrics1':checkMetrics1,'mse':mse})

    if "mae" in request.POST:
        mae=mean_absolute_error(y_test, predictval)
        return render(request,'index.html',{'checkMetrics1':checkMetrics1,'mae':mae})

    if "r2score" in request.POST:
        r2score=r2_score(y_test, predictval)
        return render(request,'index.html',{'checkMetrics1':checkMetrics1,'r2score':r2score})


def checkMetrics2(request):
    checkMetrics2=True
    data=request.session['df']
    y_axis=request.session['y-axis']
    testsize=request.session['test_size']
    df = pd.read_json(data)
    X=df.drop(y_axis,axis='columns')
    y=df[y_axis]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=testsize, random_state=20)
    algo=request.session['algo']
    if algo=="AdaBoostClassifier":
        ada=AdaBoostClassifier()
        ada.fit(X_train,y_train)
        pred=ada.predict(X_test)
    elif algo=="GradientBoostingClassifier":
        grad=GradientBoostingClassifier()
        grad.fit(X_train,y_train)
        pred=grad.predict(X_test)
    elif algo=="BaggingClassifier":
        bag=BaggingClassifier()
        bag.fit(X_train,y_train)
        pred=bag.predict(X_test)
    elif algo=="ExtraTreesClassifier":
        extraTree=ExtraTreesClassifier()
        extraTree.fit(X_train,y_train)
        pred=extraTree.predict(X_test)
    elif algo=="DecisionTreeClassifier":
        dec=DecisionTreeClassifier()
        dec.fit(X_train,y_train)
        pred=dec.predict(X_test)
    elif algo=="RadiusNeighborsClassifier":
        radius=RadiusNeighborsClassifier()
        radius.fit(X_train,y_train)
        pred=radius.predict(X_test)    
    elif algo=="KNeighborsClassifier":
        kneigh=KNeighborsClassifiern()
        kneigh.fit(X_train,y_train)
        pred=kneigh.predict(X_test)
    elif algo=="Logistics":
        logi=Logistics()
        logi.fit(X_train,y_train)
        pred=logi.predict(X_test)      
    predictval=pred
    
    if "F1 Score" in request.POST:
        f1=f1_score(y_test, predictval)
        return render(request,'index.html',{'checkMetrics2':checkMetrics2,'f1':f1})
 
    if "F BetaScore" in request.POST:
        fbeta=fbeta_score(y_test, predictval)
        return render(request,'index.html',{'checkMetrics2':checkMetrics2,'fbeta':fbeta})

    if "Log loss" in request.POST:
        log=log_loss(y_test, predictval , eps=1e-15)
        return render(request,'index.html',{'checkMetrics2':checkMetrics2,'log':log})

    if "Recall Score" in request.POST:
        rec_macro=recall_score(y_true, y_pred, average='macro')
        rec_micro=recall_score(y_true, y_pred, average='micro')
        rec_weigh=recall_score(y_true, y_pred, average='weighted')
        return render(request,'index.html',{'checkMetrics2':checkMetrics2,'rec_macro':rec_macro,'rec_micro':rec_micro,'rec_weigh':rec_weight})

    if "Accuracy Score" in request.POST:
        acc=accuracy_score(y_test, predictval)
        return render(request,'index.html',{'checkMetrics2':checkMetrics2,'acc':acc})

    if "Confusion Matrix" in request.POST:
        con=confusion_matrix(y_test, predictval)
        return render(request,'index.html',{'checkMetrics2':checkMetrics2,'con':con})

    if "ROC AUC Score" in request.POST:
        roc=roc_auc_score(y_test, predictval)
        return render(request,'index.html',{'checkMetrics2':checkMetrics2,'roc':roc})