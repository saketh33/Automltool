import numpy as np
import pandas as pd
df=pd.read_csv("C:\\Users\\saketh\\Documents\\Pune_house_data.csv")
df=df.drop('society',axis='columns')
df=df.dropna()
df['bhk']=df['size'].apply(lambda x:int(x.split(' ')[0]))
df=df.drop('size',axis='columns')
def isFloat(x):
    try:
        float(x)
    except:
        return False
    return True
df[~df['total_sqft'].apply(isFloat)]
def convert_sqft_to_num(x):
    tokens=x.split('-')
    if(len(tokens)==2):
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
df['new_total_sqft']=df.total_sqft.apply(convert_sqft_to_num)
df=df.drop('total_sqft',axis='columns')
df=df.dropna()
df1=df.copy()
df1['price_per_sqft']=(df1['price']*100000)/df1['new_total_sqft']
locations=list(df['site_location'].unique())
df1.site_location=df1.site_location.apply(lambda x: x.strip())
location_stats=df1.groupby('site_location')['site_location'].agg('count').sort_values(ascending=False)
locations_less_than_10=location_stats[location_stats<=10]
df1.site_location=df1.site_location.apply(lambda x: 'other' if x in locations_less_than_10 else x)
dates=df1.groupby('availability')['availability'].agg('count').sort_values(ascending=False)
dates_not_ready=dates[dates<10000]
df1.availability=df1.availability.apply(lambda x: 'Not Ready' if x in dates_not_ready else x)
df2=df1[~(df1.new_total_sqft/df1.bhk<300)]
def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,sub_df in df.groupby('site_location'):
        m=np.mean(sub_df.price_per_sqft)
        sd=np.std(sub_df.price_per_sqft)
        reduce_df=sub_df[(sub_df.price_per_sqft>(m-sd)) & (sub_df.price_per_sqft<(m+sd))]
        df_out=pd.concat([df_out,reduce_df],ignore_index=True)
    return df_out
df3=remove_pps_outliers(df2)
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for site_location, site_location_df in df.groupby('site_location'):
        bhk_stats={}
        for bhk, bhk_df in site_location_df.groupby('bhk'):
            bhk_stats[bhk]= {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in site_location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')
df4 = remove_bhk_outliers(df3)
df5 = df4[df4.bath<(df4.bhk+2)]
df6 = df5.copy()
df6 = df6.drop('price_per_sqft', axis='columns')
dummy_cols = pd.get_dummies(df6.site_location)
df6 = pd.concat([df6,dummy_cols], axis='columns')
dummy_cols = pd.get_dummies(df6.availability).drop('Not Ready', axis='columns')
df6 = pd.concat([df6,dummy_cols], axis='columns')
dummy_cols = pd.get_dummies(df6.area_type).drop('Super built-up  Area',axis='columns')
df6 = pd.concat([df6,dummy_cols], axis='columns')
df6.drop(['area_type','availability','site_location'], axis='columns', inplace=True)


X=df6.drop('price',axis='columns')
y=df6['price']
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.20, random_state=20)
def heck(sa,pr,sr,pk,si):
    a=sa
    b=pr
    mod=sr
    def fit(a,b,mod):
        model=mod
        f=model.fit(a,b)
        return f
    def predict(c):
        pred=mod.predict(c)
        return pred
    def score(x,y):
        return r2_score(x,y)
    mod=fit(a,b,mod)
    y_pred=predict(pk)
    return score(si,y_pred)

print("Hello welcome to techsnap automl");print("Please make sure that your dataset is cleaned and doesnt contain any categorical values");n=int(input("enter your problem type:\n1.Regression \n2.Classification\n3.exit\n"))
while(n!=3):
    if n==1:
        print("you had selceted regression");algo=input("enter your required algo name or code\n")
        if algo=="linear":
            print(heck(X_train,y_train,LinearRegression(),X_test,y_test))
        elif algo=="decision":
            print(heck(X_train,y_train,DecisionTreeRegressor(),X_test,y_test))
        elif algo=="Random":
            print(heck(X_train,y_train,RandomForestRegressor(),X_test,y_test))
        elif algo=="logistic":
            print(heck(X_train,y_train,LogisticRegression(),X_test,y_test))
        elif algo=="svm":
            print(heck(X_train,y_train,SVR(),X_test,y_test))
        elif algo=="knn":
            print(heck(X_train,y_train,KNeighborsRegressor(),X_test,y_test))
        elif algo=="ridge":
            print(heck(X_train,y_train,Ridge(),X_test,y_test))
        elif algo=="all":
            print("linear",heck(X_train,y_train,LinearRegression(),X_test,y_test));print("decision",heck(X_train,y_train,DecisionTreeRegressor(),X_test,y_test));print("Random",heck(X_train,y_train,RandomForestRegressor(),X_test,y_test));print("svm",heck(X_train,y_train,SVR(),X_test,y_test));print("knn",heck(X_train,y_train,KNeighborsRegressor(),X_test,y_test));print("ridge",heck(X_train,y_train,Ridge(),X_test,y_test))
        elif algo=="exit":
            break;
        else:
            print("selected wrong code or check the algo name")
    elif n==2:
        print("you had selected classification");algo=input("enter your required algo name or code\n")
        if algo=="decision":
            print(heck(X_train,y_train,DecisionTreeRegressor(),X_test,y_test))
        elif algo=="Random":
            print(heck(X_train,y_train,RandomForestRegressor(),X_test,y_test))
        elif algo=="logistic":
            print(heck(X_train,y_train,LogisticRegression(),X_test,y_test))
        elif algo=="svm":
            print(heck(X_train,y_train,SVR(),X_test,y_test))
        elif algo=="knn":
            print(heck(X_train,y_train,KNeighborsRegressor(),X_test,y_test))
        elif algo=="ridge":
            print(heck(X_train,y_train,Ridge(),X_test,y_test))
        elif algo=="all":
            print("linear",heck(X_train,y_train,LinearRegression(),X_test,y_test));print("decision",heck(X_train,y_train,DecisionTreeRegressor(),X_test,y_test));print("Random",heck(X_train,y_train,RandomForestRegressor(),X_test,y_test));print("svm",heck(X_train,y_train,SVR(),X_test,y_test));print("knn",heck(X_train,y_train,KNeighborsRegressor(),X_test,y_test));print("ridge",heck(X_train,y_train,Ridge(),X_test,y_test))
        elif algo=="exit":
            break;
        else:
            print("selected wrong code or check the algo name")




from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
mse= mean_squared_error(expected, predicted)
rmse= mean_squared_error(expected, predicted, squared=False)
mae= mean_absolute_error(expected, predicted)
r2score= r2_score(y_true, y_pred)


import xgboost as xg
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor

adar= AdaBoostRegressor(random_state=0, n_estimators=100)
catr= CatBoostRegressor(iterations=2,learning_rate=1,depth=2)
gradr= GradientBoostingRegressor(random_state=0)
xgbr= xg.XGBRegressor(objective ='reg:linear',n_estimators = 10, seed = 123)

Adar.score


from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier


from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import log_loss 
log_loss(y_true, y_pred, eps=1e-15)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
recall_score(y_true, y_pred, average='macro')
recall_score(y_true, y_pred, average='micro')
recall_score(y_true, y_pred, average='weighted')
 
