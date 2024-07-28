import streamlit as st
import pandas as pd
import requests
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain  ## for question answering prompts
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser


import os
import google.generativeai as genai
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

api_key=os.getenv("api_key")

st.set_page_config(
    page_title='Movie Recommender'
)

# model=genai.GenerativeModel("gemini-pro")

st.title("Movie Recommendation System")

st.write("This is a movie recommender app")

input_movie_name = st.text_input("Enter the movie name")


def movie_img(movie_name):
    base_url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": api_key,
        "query": movie_name
    }
    response = requests.get(base_url, params=params)
    data = response.json()

    if response.status_code == 200 and data.get("results"):
        movie_info = data['results'][0]
        poster_path = movie_info.get("poster_path")
        if poster_path:
            base_image_url = "http://image.tmdb.org/t/p/w500/"
            return base_image_url + poster_path
        else:
            print(f"Poster path not found for '{movie_name}'.")
            return None
    else:
        print(f"Error fetching movie information for '{movie_name}'.")
        st.error(f"Error: Could not find movie information for '{input_movie_name}'.")
        return None


def recommend(movie):
    prompt = ChatPromptTemplate.from_template("""
        You are a movie recommender bot. Give 5 movie recommendations(only names) similar to {movie}.
        Provide output without numbering(bullets).
    """)
    llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                 temperature=0.3)

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    try:
        responses = chain.invoke({"movie": input_movie_name})
        recommendations = responses.strip().split('\n')  # Split by newline for multiple recommendations
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        # st.error("Error: Could not generate movie recommendations.")
        return None

    recommended_movies = []
    for movie_name in recommendations:
        poster_url = movie_img(movie_name)
        if poster_url:
            recommended_movies.append((movie_name, poster_url))
        else:
            print(f"Poster not found for '{movie_name}'.")

    return recommended_movies


if st.button("Recommend"):
    st.spinner(text="Searching...")
    recommended_movies = recommend(input_movie_name)

    if recommended_movies:
        num_recommendations = len(recommended_movies)
        st.text(f"Recommendations for '{input_movie_name}':")
        cols = st.columns(min(num_recommendations, 5))  # Create up to 3 columns

        for i, (name, poster_url) in enumerate(recommended_movies):
            with cols[i]:
                st.image(poster_url)
                st.write(name)
                
    else:
        st.warning("No recommendations found.")

