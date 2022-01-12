import pandas as pd
import streamlit as st
import numpy as np
import requests
from matplotlib import pyplot as plt
import numpy as np
import os
# import seaborn as sns
# sns.set()

api_url = "https://banking-data-dashboard.herokuapp.com/"

main_features_pd = pd.read_csv("main_features_pd.csv", index_col="index")
X_train2_sc_pd_mean = pd.read_csv("X_train2_sc_pd_mean.csv", index_col="index")
sample = pd.read_csv("X_test2_sc_pd_sample.csv", index_col="index")

def request_prediction( model_uri, data):

    request = requests.post(model_uri, data)

    if request.status_code != 200:
        raise Exception("Request failed with status {}, {}".format(request.status_code, request.text))

    return request.json()

class Welcome_page:

    def app():
        st.header("Welcome Page")

        text = """
        This dashboard was built for the project 7 of the data science course of [https://openclassrooms.com/fr/paths/164-data-scientist](Openclassrooms)
        
        The purposes of the dashboard is to :  
        + Present the key aspects of the given data set.
        + Display the data of a list of clients to evaluate whether or not the bank should grant them a loan.
        + Evaluate the solvability of a dummy client with data you input.
            
        The probability are given by a rest api hosted on heroku : [https://banking-data-dashboard.herokuapp.com/](https://banking-data-dashboard.herokuapp.com/)  
        The model is the following : 
        > LGBMClassifier( max_depth=6,  
                        min_child_samples=500,  
                        min_child_weight=0.05,  
                        min_split_gain=0.05,  
                        num_leaves=48)  
        
        You can find the code at : [https://github.com/Neclis/OC7](https://github.com/Neclis/OC7)
        """
        st.markdown(text)

class Data_overview:

    def app():
        st.header('Data Overview')
        st.write('This page presents some key features of the data and the ones that are the most impactful for the model.')
        for file in os.listdir(path='./pics'):
            st.image('./pics/'+file)
        
class App_prediction_from_id:

    def app():
        # FLASK_URL = "http://127.0.0.1:5000/enterid"
        FLASK_URL = api_url+"/enterid"

        st.header('Solvability Prediction by id')
        
        client_ID = st.selectbox("Client ID", sample.index)

        results = pd.DataFrame(sample.loc[client_ID, main_features_pd.index].values, index=main_features_pd.index, columns=["data"])
        st.bar_chart(results)

        col1, col2 = st.columns(2)
        
        for i , var in enumerate(main_features_pd.index):
            fig, ax = plt.subplots(1,1, figsize = (6,6))
            ax.set_title("Client data : {}".format(round(sample.loc[client_ID, var])))
            ax.hist(sample[var])
            ax.axvline(x = sample.loc[client_ID, var] , c= "r", linewidth=4)

            if i < 5:
                col1.write(var)
                col1.pyplot(fig)
            else:
                col2.write(var)
                col2.pyplot(fig)

        my_placeholder = st.empty()
        col1, col2 = st.columns(2)
        my_placeholder1 = col1.empty()
        my_placeholder2 = col2.empty()
        my_placeholder3 = st.empty()
        predict_btn = st.button('Predict')

        if predict_btn:
            data_json = {'ide': client_ID}
            pred = request_prediction(FLASK_URL, data_json)
            
            my_placeholder.write("The lightgbm model predicts the following probabilities:")
            my_placeholder1.metric("Not solvable", "{} %".format(round(pred["proba_0"],2)))
            my_placeholder2.metric("Solvable", "{} %".format(round(pred["proba_1"],2)))

            fig, ax = plt.subplots(1,1, figsize = (4,4))
            fig.suptitle("Customer {}".format(client_ID))
            ax.set_title("The probability of solvability is {} %".format(round(pred["proba_1"],2)))
            ax.pie(list(pred.values()), labels= list(pred.keys()) , colors=["r","g"])
            ax.axis("square")

            my_placeholder3.pyplot(fig)

class App_prediction_from_data:

    def app():
        # FLASK_URL = "http://127.0.0.1:5000/enterdata"
        FLASK_URL = api_url+"/enterdata"

        st.header('Solvability Prediction from data input')
        
        st.write("Here are the main values and their range :")
        st.dataframe(main_features_pd[["Min","Mean","Med","Max"]])
        
        st.write("Adjust the sliders to get your prediction :")

        data = np.zeros(len(main_features_pd.index))
        
        col1, col2 = st.columns(2)   

        for i,var in enumerate(main_features_pd.index):
            if i < 5:
                data[i] = col1.slider(label = var, min_value=round(main_features_pd.loc[var,"Min"].item(),2), value=round(main_features_pd.loc[var,"Med"].item(),2), max_value=round(main_features_pd.loc[var,"Max"].item(),2), step=0.01)
                
                fig, ax = plt.subplots(1,1, figsize = (3,3))
                ax.hist(sample[var])
                ax.axvline(x = data[i] , c= "r", linewidth=4)
                col1.pyplot(fig)

            else :
                data[i] = col2.slider(label = var, min_value=round(main_features_pd.loc[var,"Min"].item(),2), value=round(main_features_pd.loc[var,"Med"].item(),2), max_value=round(main_features_pd.loc[var,"Max"].item(),2), step=0.01)
                
                fig, ax = plt.subplots(1,1, figsize = (3,3))
                ax.hist(sample[var])
                ax.axvline(x = data[i] , c= "r", linewidth=4)
                col2.pyplot(fig)

        my_placeholder = st.empty()
        col1, col2 = st.columns(2)
        my_placeholder1 = col1.empty()
        my_placeholder2 = col2.empty()
        my_placeholder3 = st.empty()
            
        predict_btn = st.button('Prédire')

        if predict_btn:
            
            data_json = {main_features_pd.index[i] : data[i] for i in range(len(main_features_pd.index))}
            pred = request_prediction(FLASK_URL, data_json)
                        
            my_placeholder.write("The lightgbm model predicts the following probabilities:")
            my_placeholder1.metric("Not solvable", "{} %".format(round(pred["proba_0"],2)))
            my_placeholder2.metric("Solvable", "{} %".format(round(pred["proba_1"],2)))

            fig, ax = plt.subplots( 1,1, figsize = (4,4))
            ax.set_title("The probability of solvability is {} %".format(round(pred["proba_1"],2)))
            ax.pie(list(pred.values()), labels= list(pred.keys()) , colors=["r","g"])
            ax.axis("square")

            my_placeholder3.pyplot(fig)

#### Main function of the App
def main():
    
    st.title('Loan Dashboard')
    st.write("Welcome to this Customer financial data inquiry Dashboard.")

    PAGES = {
            "Welcome page": Welcome_page,
            "Dataset overview" : Data_overview,
            "Prediction from customer ID": App_prediction_from_id,
            "Prediction from data input" : App_prediction_from_data
            }
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.app()

    st.caption("Made by Nicolas Martinez studying with OpenClassrooms. (01/2022)")


if __name__ == '__main__':
    main()

# CLI : 
# streamlit run dashboard.py