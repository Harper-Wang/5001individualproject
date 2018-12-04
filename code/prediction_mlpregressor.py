# -*- coding: utf-8 -*-
"""
Prediction of time consumed by a classfication task using MLPRegressor
"""
#%%
# load packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

traindata = pd.read_csv('.../data/train.csv') # you may have to change this path
# print(traindata.head())
testdata = pd.read_csv('.../data/test.csv') # you may have to change this path
# print(testdata.head())
randomstate = 12
modelrandomstate = 2
# feature engineering 
def transform1(x):
    # drop some irrevelant features according to feature description on website: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
    x = x.drop(columns=['random_state','scale'])

    # categorical feature encode
    penalty_mapping = {'none':0, 'l2':2, 'l1':1, 'elasticnet':3} 
    penalty_encoded = x['penalty'].map(penalty_mapping)  # mapping categories to numbters
    x['penalty_num'] = pd.Series(penalty_encoded)
    x.drop(columns=['penalty'],inplace=True)
    x['n_jobs']=[16 if m==-1 else m for m in x['n_jobs']] # change n_jbos=-1 to 16
    return x
#%%

#%%
def create_features(x):
    # create features based on feature combinations
    x['n_job_-1'] = pd.Series(1/x['n_jobs'].values)
    x['n_features_n_samples'] = pd.Series(x['n_features'].values*x['n_samples'].values)
    x['max_iter_n_samples_n_featues'] = pd.Series(x['max_iter'].values*x['n_features'].values*x['n_samples'].values)
    x['max_iter_n_samples_n_featues_n_jobs'] = pd.Series(x['max_iter'].values*x['n_features'].values*x['n_samples'].values/
                                                        x['n_jobs'].values)
    x['max_iter_n_samples_n_featues_n_classes_n_jobs'] = pd.Series(x['max_iter'].values*x['n_features'].values*x['n_samples'].values
                                                                    *x['n_classes'].values/x['n_jobs'].values)
    x['max_iter_n_samples_n_featues_n_classes_flip_n_jobs'] = pd.Series(x['max_iter'].values*x['n_features'].values*x['n_samples'].values
                                                                    *x['n_classes'].values*x['flip_y'].values/x['n_jobs'].values)
    
    x['penaly_max_iter_n_samples_n_featues_n_jobs'] = pd.Series(x['penalty_num'].values*x['max_iter'].values*x['n_features'].values*x['n_samples'].values/
                                                        x['n_jobs'].values)
    x['l1ratio_max_iter_n_samples_n_featues_n_classes_flip_n_jobs'] = pd.Series(x['l1_ratio'].values*x['max_iter'].values*x['n_features'].values*x['n_samples'].values
                                                                    *x['n_classes'].values*x['flip_y'].values/x['n_jobs'].values)
    x['penaly_max_iter2_n_samples_n_featues_n_jobs'] = pd.Series(x['penalty_num'].values*x['max_iter'].values*x['max_iter'].values*x['n_features'].values*x['n_samples'].values/
                                                        x['n_jobs'].values)
    x['penaly2_max_iter2_n_samples_n_featues_n_jobs'] = pd.Series(x['penalty_num'].values*x['penalty_num'].values*x['max_iter'].values*x['max_iter'].values*x['n_features'].values*x['n_samples'].values/
                                                        x['n_jobs'].values)
    x['penaly2_max_iter2_n_samples_n_featues_n_jobs_n_informative'] = pd.Series(x['penalty_num'].values*x['penalty_num'].values*x['max_iter'].values*x['max_iter'].values*x['n_features'].values*x['n_samples'].values*x['n_informative'].values/
                                                        x['n_jobs'].values)
    x['penaly2_max_iter2_n_samples_n_featues_n_jobs_n_informative_n_classes'] = pd.Series(x['penalty_num'].values*x['penalty_num'].values*x['max_iter'].values*x['max_iter'].values*x['n_features'].values*x['n_samples'].values*x['n_informative'].values *x['n_classes'].values/x['n_jobs'].values)
     
    
    # create features using square function, square root function, cube function                                                                                     
    for col in x.columns:
        x[col+'_2'] = pd.Series(x[col].values*x[col].values)
        x[col+'_0.5'] = pd.Series(np.sqrt(x[col].values))
        x[col+'_3'] = pd.Series(x[col].values*x[col].values*x[col].values)
#         x[col+'_log'] = pd.Series(np.log(x[col].values+1))

    return x
#%%
# transform train data and test data together
train = transform1(traindata)
train_X = create_features(train.drop(columns=['id','time']))
test = transform1(testdata)
test_X = create_features(test.drop(columns=['id']))
train_X_array = train_X.values
train_y = train['time'].values
test_X_array = test_X.values

# min-max scaling
scaler = MinMaxScaler()
train_X_num = scaler.fit_transform(train_X_array)
test_X_num = scaler.fit_transform(test_X_array)

# feature selection using RFE, and the estimator is RandomForestRegressor
rfecv = RFECV(estimator=RandomForestRegressor(n_estimators=100), step=1, cv=5)
rfecv.fit(train_X_num, train_y)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score ")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show() # draw a plot illustrating relationship of model performance and number of features

#%%
# feature_position = np.where(rfecv.ranking_<=1)[0] # choose features whose tanking<=1
feature_position = [  5,  10, 12,  15,  16,  22,  63,  64,  65,  75,   78,  83,  84,  85,  86,  89,  91,  94,
 95,  99, 102, 111, 112, 114]  # this is best position I obtained
train_X_selected = train_X_num[:,feature_position] 
test_X_selected=test_X_num[:,feature_position] 
print(feature_position)
print(train_X_selected.shape)
print(train_X.iloc[:,feature_position].columns)

# [  5  10  12  15  16  22  63  64  65  75  78  83  84  85  86  89  91  94
#   95  99 102 111 112 114]
# (400, 24)
# Index(['n_features', 'penalty_num', 'n_features_n_samples',
#        'max_iter_n_samples_n_featues_n_classes_n_jobs',
#        'max_iter_n_samples_n_featues_n_classes_flip_n_jobs',
#        'penaly2_max_iter2_n_samples_n_featues_n_jobs_n_informative_n_classes',
#        'penalty_num_2', 'penalty_num_0.5', 'penalty_num_3',
#        'max_iter_n_samples_n_featues_2', 'max_iter_n_samples_n_featues_log',
#        'max_iter_n_samples_n_featues_n_classes_n_jobs_2',
#        'max_iter_n_samples_n_featues_n_classes_n_jobs_0.5',
#        'max_iter_n_samples_n_featues_n_classes_n_jobs_3',
#        'max_iter_n_samples_n_featues_n_classes_n_jobs_log',
#        'max_iter_n_samples_n_featues_n_classes_flip_n_jobs_3',
#        'penaly_max_iter_n_samples_n_featues_n_jobs_2',
#        'penaly_max_iter_n_samples_n_featues_n_jobs_log',
#        'l1ratio_max_iter_n_samples_n_featues_n_classes_flip_n_jobs_2',
#        'penaly_max_iter2_n_samples_n_featues_n_jobs_2',
#        'penaly_max_iter2_n_samples_n_featues_n_jobs_log',
#        'penaly2_max_iter2_n_samples_n_featues_n_jobs_n_informative_n_classes_2',
#        'penaly2_max_iter2_n_samples_n_featues_n_jobs_n_informative_n_classes_0.5',
#        'penaly2_max_iter2_n_samples_n_featues_n_jobs_n_informative_n_classes_log'],
#       dtype='object')

#%%

# model selection and training 
# tune parameters using gridsearchcv 
from sklearn.neural_network import MLPRegressor
m= []
for i in range(500,1100,100):
    for j in range(10,110,10):
        for h in range(5,20,2):
            for l in range(4,15,2):
                 m.append((i,j,h,l))   
# print(m)     
girdparam = {'hidden_layer_sizes': m, 'alpha': [0.00001,0.0001,0.001,0.01]}
mlp =MLPRegressor( activation='relu',solver='lbfgs', learning_rate_init=0.1,random_state=randomstate, alpha=0.0001,max_iter=2000)
randomsearch = RandomizedSearchCV(mlp, girdparam, cv=5, random_state=modelrandomstate)
randomsearch.fit(train_X_selected, train_y)
print(randomsearch.best_params_)
print(randomsearch.best_score_)
# {'hidden_layer_sizes': (600, 90, 9, 6), 'alpha': 0.001}
# 0.9415434360366618


#%%
# build final model using best parameters
mlp2 =MLPRegressor( hidden_layer_sizes=(600,90,9,6), activation='relu',solver='lbfgs'
, random_state=randomstate,learning_rate_init=0.00001, max_iter=2000, alpha=0.001)
mlp2.fit(train_X_selected, train_y)
predictions_mlp = mlp2.predict(test_X_selected) # predict
# print(predictions_mlp) 
df=pd.DataFrame()
df['Id'] = range(100)
df['time'] = predictions_mlp 
display(df.head())
df.to_csv('.../results/result_mlp.csv', index=None) # write results to csv, change this path to your own


#%%

