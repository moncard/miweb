import streamlit as st 
import pandas as pd 
import numpy as np
import pickle

pickle_in = open("lgbm_housing.pkl","rb")
classifier=pickle.load(pickle_in)

df = pd.read_csv('app_test_domain.csv')

def classify(prediction):
    if prediction == 0: 
        return ' class 0' 
    elif prediction ==1:
        return 'class 1'
    else: 
        return 'error'

def main():
    st.title('Outil de Prediction')
    st.sidebar.header('Inserer ID')
    #
    def user_input_parameters():
        id = st.sidebar.number_input('inserer le id',min_value=100001,
                                    max_value=456250,label_visibility='collapsed')
        return id

    x=df[df['SK_ID_CURR']==user_input_parameters()]
    st.subheader('Dataframe')
    st.write(df)

    if st.button('RUN'):
        st.success(classify(classifier.predict(x)))

if __name__ == '__main__':
    main()