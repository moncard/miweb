

import pandas as pd
import pickle
import uvicorn
# Api import
from pydantic import BaseModel
from fastapi import FastAPI
from fonctions import *

app = FastAPI()
pickle_in = open("lgbm_housing.pkl","rb")
classifier=pickle.load(pickle_in)
app_test=pd.read_csv('app_test_domain.csv')

pickle_df = open('df_test.pkl','rb')
df = pickle.load(pickle_df)
"""
#test = app_test.copy()
submit = app_test[['SK_ID_CURR']]
test = test.drop(columns = ['SK_ID_CURR'])

test=imputer(test,SimpleImputer(strategy = 'median'))
test=imputer(test,MinMaxScaler(feature_range = (0, 1)))
df= pd.DataFrame(test)
df['SK_ID_CURR']=submit
"""
@app.get("/")
def read_root():
    ##y_predict=loaded_model.predict(X_test)
    return {"Hello": "world"}

@app.get("/model/{id}")
def read_item(id: int):
    id_test=df[df['SK_ID_CURR']==id]
    prediction = classifier.predict(id_test)
    print(prediction)
    return {"item_id": 'prediction'}


    