import pickle
import pandas as pd
import numpy as np


class SentimentRecommenderSystem:
    root_model_path = "pickles/"
    sentiment_model = "logistic_final_model.pkl"
    tfidf_vectorizer = "tfidf.pkl"
    best_recommender = "best_recommendation_model.pkl"
    sent_dataframe = "sentiment_dataframe.pkl"

    def __init__(self):
        self.sentiment_model = pickle.load(open(
            SentimentRecommenderSystem.root_model_path + SentimentRecommenderSystem.sentiment_model, 'rb'))
        self.tfidf_vectorizer = pd.read_pickle(
            SentimentRecommenderSystem.root_model_path + SentimentRecommenderSystem.tfidf_vectorizer)
        self.user_final_rating = pickle.load(open(
            SentimentRecommenderSystem.root_model_path + SentimentRecommenderSystem.best_recommender, 'rb'))
        self.snmt_data = pickle.load(open(
            SentimentRecommenderSystem.root_model_path + SentimentRecommenderSystem.sent_dataframe, 'rb'))

    def top5_recommendations(self, user_name):
        if user_name not in self.user_final_rating.index:
            print(f"The User {user_name} does not exist. Please provide a valid user name")
            return None
        else:
            # Get top 20 recommended products from the best recommendation model
            top20_recommended_products = list(
                self.user_final_rating.loc[user_name].sort_values(ascending=False)[0:20].index)
            # Get only the recommended products from the prepared dataframe "df_sent"
            df_top20_products = self.snmt_data[self.snmt_data.id.isin(top20_recommended_products)]
            # For these 20 products, get their user reviews and pass them through TF-IDF vectorizer to convert the data into suitable format for modeling
            X = self.tfidf_vectorizer.transform(df_top20_products["lemmatized_reviews"].values.astype(str))
            # Use the best sentiment model to predict the sentiment for these user reviews
            df_top20_products['predicted_sentiment'] = self.sentiment_model.predict(X)
            # Create a new dataframe "pred_df" to store the count of positive user sentiments
            pred_df = pd.DataFrame(df_top20_products.groupby(by='name').sum()['predicted_sentiment'])
            pred_df.columns = ['pos_snmt_count']
            # Create a column to measure the total sentiment count
            pred_df['total_snmt_count'] = df_top20_products.groupby(by='name')['predicted_sentiment'].count()
            # Create a column that measures the % of positive user sentiment for each product review
            pred_df['positive_snmt_percentage'] = np.round(pred_df['pos_snmt_count'] / pred_df['total_snmt_count'] * 100, 2)
            # Return top 5 recommended products to the user
            result = list(pred_df.sort_values(by='positive_snmt_percentage', ascending=False)[:5].index)
            return result
