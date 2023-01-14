import streamlit as st
from model import SentimentRecommenderSystem
import pandas as pd


sent_reco_model = SentimentRecommenderSystem()

st.title("Sentiment Based Product Recommendation System")

with st.form(key='my_form'):
	text_input = st.text_input(label='Enter Username')
	submit_button = st.form_submit_button(label='Submit')
if submit_button:
    # Get the username as input
    user_name_input = text_input.lower()
    sent_reco_output = sent_reco_model.top5_recommendations(user_name_input)
    if str(type(sent_reco_output))=="<class 'NoneType'>":
        st.write("Invalid Username. Please try again")
    else:
        st.subheader('Top 5 Product Recommendations')
        result_df=pd.DataFrame({"S.No":[1,2,3,4,5],"ProductName":sent_reco_output})
        st.dataframe(data=result_df,use_container_width=True)  

st.subheader('Developed by Bhanu Prakash Doppalapudi')        
