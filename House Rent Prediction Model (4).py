#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[415]:


df = pd.read_excel("House_Rent_Train.xlsx")


# In[416]:


df.head()


# In[417]:


#column id does not help in determining with the home price so we are droping that column
df.drop('id',axis= 1,inplace = True)


# In[418]:


df.drop('activation_date',axis= 1,inplace = True)


# In[419]:


df.nunique()


# In[420]:


#checking if there are any null values 
print(df.isnull().sum())


# In[421]:


#droping the null values 
df = df.dropna()


# In[422]:


import json


def parse_amenities(amenities_str):
    amenities_dict = json.loads(amenities_str)
    for key, value in amenities_dict.items():
        if value == "true":
            amenities_dict[key] = True
        elif value == "false":
            amenities_dict[key] = False
    return amenities_dict

df['amenities'] = df['amenities'].apply(parse_amenities)


# In[423]:


notnull(df['amenities'] = df['amenities'].map(lambda x: True if x == 'true' and pd.notnull(x) else x is False if x == 'false' and pd.x) else x)


# In[424]:


df1 = pd.concat([df, pd.json_normalize(df.pop("amenities"))], axis=1)


# In[425]:


df1


# In[426]:


print(df1.isnull().sum())


# In[427]:


correlation_matrix = df1.corr()


# In[428]:


correlation_matrix


# In[429]:


df1


# In[430]:


plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap between Features and Target')
plt.show()


# In[433]:


target_corr = correlation_matrix['rent'].drop('rent')  # Replace 'target_variable' with your actual target column name
features = target_corr.index

plt.figure(figsize=(8, 6))
plt.barh(features, target_corr)
plt.xlabel('Correlation')
plt.ylabel('Features')
plt.title('Correlation between Features and Target')
plt.show()


# In[434]:


df1.columns


# In[435]:


df1.drop('LIFT',axis= 1,inplace = True)


# In[436]:


df1.drop('GYM',axis= 1,inplace = True)


# In[437]:


df1.info()


# In[438]:


object_columns = df1.select_dtypes(include=['object']).columns


# In[439]:


object_columns


# In[440]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[441]:


for col in object_columns:
    df1[col] = le.fit_transform(df1[col])


# In[442]:


df1


# In[443]:


print(df1.isnull().sum())


# In[444]:


df1 = df1.dropna()


# In[445]:


df1


# In[446]:


correlation_matrix = df1.corr()


# In[447]:


plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap between Features and Target')
plt.show()


# In[410]:


target_corr = correlation_matrix['rent'].drop('rent')  # Replace 'target_variable' with your actual target column name
features = target_corr.index

plt.figure(figsize=(20, 18))
plt.barh(features, target_corr)
plt.xlabel('Correlation')
plt.ylabel('Features')
plt.title('Correlation between Features and Target')
plt.show()


# In[492]:


# Outlier Detection 


# In[411]:


sns.boxplot(data = df1)


# In[ ]:





# In[490]:


X = df1[df1.columns.drop('rent')]
Y = df1['rent']


# In[449]:


#Y = Y.to_numpy()


# In[452]:


Y.describe()


# #scaling the data 

# In[453]:


from sklearn.preprocessing import StandardScaler


# In[454]:


sc = StandardScaler()
cols = X.columns
X[cols] = sc.fit_transform(X[cols])


# In[217]:


#X = X.to_numpy()


# In[455]:


X.head()


# # Basic Regression Model

# In[456]:


from sklearn.linear_model import LinearRegression


# In[457]:


model = LinearRegression()


# In[458]:


model.fit(X,Y)


# In[459]:


model.score(X,Y)


# In[491]:


#Multicolinaearity Check


# In[460]:


import statsmodels.api as sm


# In[461]:


X_sm = sm.add_constant(X)
sm_model = sm.OLS(Y,X_sm).fit()


# In[462]:


print(sm_model.summary())


# In[463]:


#checking for multicolinearity


# In[464]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[465]:


vif = pd.DataFrame()


# In[466]:


x_t = X
vif['Features'] = x_t.columns
vif['VIF'] = [variance_inflation_factor(x_t.values,i) for i in range(x_t.shape[1])]


# In[467]:


vif['VIF'] = round(vif['VIF'],2)


# In[468]:


vif = vif.sort_values(by = "VIF",ascending= False)


# In[469]:


vif


# In[470]:


sns.displot(Y)
#the target variable is not following normal distribution


# In[471]:


pred = model.predict(X)


# In[472]:


errors = pred - Y


# In[473]:


sns.displot(errors)


# In[474]:


sns.scatterplot(pred,errors)


# In[242]:


Y.describe()


# In[243]:


#df2 = df1[df1['rent']<40000]


# In[476]:


#df2


# In[475]:


#sns.displot(df2['rent'])


# In[249]:


#X1 = df2[df2.columns.drop('rent')]
#Y1 = df2['rent']


# In[477]:


#sc = StandardScaler()
#cols = X1.columns
#X1[cols] = sc.fit_transform(X1[cols])


# In[253]:


#model1 = LinearRegression()


# In[478]:


#model1.fit(X1,Y1)


# In[479]:


#model1.score(X1,Y1)


# In[480]:


from sklearn.metrics import mean_squared_error


# In[481]:


ep = np.sqrt(mean_squared_error(pred,Y))


# In[482]:


ep


# In[485]:


from sklearn.ensemble import GradientBoostingRegressor


# In[553]:


from sklearn.model_selection import KFold
folds = KFold(n_splits=4)


# In[486]:


#hyperparameters tuning
gb_model = GradientBoostingRegressor(random_state=1)
param = {
    'n_estimators' : [50,100,150],
    'learning_rate' : [0.1,0.01,0.05],
    'max_depth' : [5,8,10]
}


# In[487]:


gscv_gb_model= GridSearchCV(gb_model,param_grid=param,cv= folds,verbose=1)


# In[488]:


from sklearn.model_selection import train_test_split,GridSearchCV
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=1)


# In[489]:


gscv_gb_model.fit(X_train,Y_train)


# In[533]:


gscv_gb_model.score(X_train,Y_train)


# In[534]:


gscv_gb_model.score(X_test,Y_test)


# #Test data

# In[506]:


new_data = pd.read_excel('House_Rent_Test.xlsx')


# In[507]:


new_data = new_data.drop('id',axis = 1)


# In[508]:


new_data = new_data.drop('activation_date',axis= 1)


# In[509]:


new_data


# In[500]:


print(new_data.isnull().sum())


# In[510]:


import json


def parse_amenities(amenities_str):
    amenities_dict = json.loads(amenities_str)
    for key, value in amenities_dict.items():
        if value == "true":
            amenities_dict[key] = True
        elif value == "false":
            amenities_dict[key] = False
    return amenities_dict

new_data['amenities'] = new_data['amenities'].apply(parse_amenities)


# In[511]:


notnull(new_data['amenities'] = new_data['amenities'].map(lambda x: True if x == 'true' and pd.notnull(x) else x is False if x == 'false' and pd.x) else x)


# In[512]:


new_data1 = pd.concat([new_data, pd.json_normalize(new_data.pop("amenities"))], axis=1)


# In[513]:


new_data1


# In[ ]:





# In[293]:


new_data


# In[294]:


new_data['amenities'] = new_data['amenities'].map(lambda x: True if x == 'true' and pd.notnull(x) else x is False if x == 'false' and pd.notnull(x) else x)


# In[295]:


new_data1 = pd.concat([new_data, pd.json_normalize(new_data.pop("amenities"))], axis=1)


# In[520]:


new_data1


# In[521]:


new_data1.columns


# In[527]:


new_data1.info()


# In[528]:


object_columns = new_data1.select_dtypes(include=['object','bool']).columns


# In[529]:


for col in object_columns:
    new_data1[col] = le.fit_transform(new_data1[col])


# In[530]:


new_data1


# In[531]:


sc = StandardScaler()
cols = new_data1.columns
new_data1[cols] = sc.fit_transform(new_data1[cols])


# In[535]:


Test_prediction = gscv_gb_model.predict(new_data1)


# In[538]:


Test_prediction


# In[552]:


#XGboost model


# In[554]:


import xgboost as xgb


# In[555]:



params = {
    'n_estimators' : 150,
    'max_depth':5,
    'eta':0.1
    
    
}
model2 = xgb.XGBRegressor(**params)


# In[557]:


model2.fit(X_train,Y_train,
           eval_set = [(X_train,Y_train),(X_test,Y_test)],
           verbose = True)


# In[558]:


model2.score(X_train,Y_train)


# In[559]:


model2.score(X_test,Y_test)


# In[560]:


test_pred_xgb = model2.predict(new_data1)


# In[561]:


test_pred_xgb


# In[562]:


solution  = "output.csv"
np.savetxt(solution, test_pred_xgb, delimiter=',', fmt='%d')


# In[ ]:




