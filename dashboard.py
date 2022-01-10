import pandas as pd
import streamlit as st
import numpy as np
import requests
# from page1 import App_prediction_from_id

api_url = "https://banking-data-dashboard.herokuapp.com/"
main_features = ['EXT_SOURCE_1',
                'EXT_SOURCE_2',
                'EXT_SOURCE_3',
                'CODE_GENDER',
                'NAME_FAMILY_STATUS',
                'AMT_REQ_CREDIT_BUREAU_TOTAL',
                'BIRTH_EMPLOYED_INTERVEL',
                'AMT_INCOME_TOTAL',
                'AMT_GOODS_PRICE',
                'AMT_CREDIT_SUM_DEBT'
                ]

main_features_pd = pd.read_pickle("main_features_pd.pickle")
X_train2_sc_pd_mean = pd.read_pickle("X_train2_sc_pd_mean.pickle")
sample = pd.read_pickle("X_test2_sc_pd_sample.pickle")

def request_prediction( model_uri, data):

    # for (key, value) in zip(data_json.keys(), data_json.values()):
    #             st.write(key,  value)

    request = requests.post(model_uri, data)

    if request.status_code != 200:
        raise Exception("Request failed with status {}, {}".format(request.status_code, request.text))

    return request.json()

class App_prediction_from_id:

    def app():
        # FLASK_URL = "http://127.0.0.1:5000/enterid"
        FLASK_URL = api_url+"/enterid"

        st.header('Solvability Prediction by id')
        
        client_ID = st.selectbox("Client ID", sample.index)
        
        st.write('Voici les données pour le client {}'.format(client_ID))

        for col in sample[main_features].columns:
            st.write('Le {} est de : {:.2f}'.format(col, sample.loc[client_ID, col]))
        
        results = pd.DataFrame(sample.loc[client_ID, main_features].values, index=main_features, columns=["data"])
        results

        st.bar_chart(results)

        predict_btn = st.button('Prédire')

        if predict_btn:
            data_json = {'ide': client_ID}
            pred = request_prediction(FLASK_URL, data_json)
            
            for (key, value) in zip(pred.keys(), pred.values()):
                st.write(key,  value)

class App_prediction_from_data:

    def app():
        # FLASK_URL = "http://127.0.0.1:5000/enterdata"
        FLASK_URL = api_url+"/enterdata"

        st.header('Solvability Prediction from data input')
        
        st.dataframe(main_features_pd[["Min","Mean","Med","Max"]])
        
        data = np.zeros(len(main_features_pd.index))
        
        col1, col2 = st.columns(2)
        for i,var in enumerate(main_features_pd.index):
            if i < 5:
                data[i] = col1.number_input(var, min_value=main_features_pd.loc[var,"Min"], value=main_features_pd.loc[var,"Med"], max_value=main_features_pd.loc[var,"Max"], step=0.01)            
            else :
                data[i] = col2.number_input(var, min_value=main_features_pd.loc[var,"Min"], value=main_features_pd.loc[var,"Med"], max_value=main_features_pd.loc[var,"Max"], step=0.01)            

        predict_btn = st.button('Prédire')

        if predict_btn:
            
            data_json = {main_features_pd.index[i] : data[i] for i in range(len(main_features_pd.index))}
            pred = request_prediction(FLASK_URL, data_json)
                        
            for (key, value) in zip(pred.keys(), pred.values()):
                st.write(key,  value)


#### Main function of the App
def main():
    
    st.title('Loan Dashboard')

    PAGES = {
            "Prediction from customer ID": App_prediction_from_id,
            "Prediction from data input" : App_prediction_from_data
            }
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.app()


if __name__ == '__main__':
    main()

# CLI : 
# streamlit run dashboard.py