

import streamlit as st
import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from PIL import Image
from langchain.document_loaders import WebBaseLoader
from newsapi import NewsApiClient
import requests
import datetime

st.set_page_config(
    page_title="Demo Page",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


url = 'https://www.scmp.com/news/hong-kong/transport/article/3227297/hongkongers-flood-hk-express-website-carrier-gives-away-21626-free-tickets-many-fail-enter-booking'

# Get the current datetime as a string
now = datetime.datetime.now()
filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"


sample_text = """
"""
news_endpoint = 'https://newsapi.org/v2/top-headlines?country=HK&apiKey=3302aaafa5ae4d9eb441f833e249ce77'
news_endpoint = 'https://newsapi.org/v2/top-headlines?sources=techcrunch&apiKey=3302aaafa5ae4d9eb441f833e249ce77'
st.subheader(f':blue[Extract content from news api endpoint]   {news_endpoint}')

response = requests.get(news_endpoint)
st.subheader('response text ')
# st.write(response.text)
output = ""
if response.status_code == 200:
    data = response.json()
    with st.expander(" Json 1"):
        st.code(data)
    with st.expander(" Json 2"):
        st.write(data)
    st.subheader('Json content')
    articles = data['articles']
    
    with open(filename, "w") as file:
        for article in articles:

            title = article['title'] 
            url = article['url']
            webLink = "\n<a href='"  +  article['url']    + "'>" +  article['title']  + "</>"
            output = output + webLink
            st.info(webLink)
            
            # get the content of the artice
            loader = WebBaseLoader([url])
            data = loader.load()
            WebContent = data[0].page_content.replace('\n', '')
            file.write("\n\n")
            file.write("[TITLE]\n" + title + "\n")
            file.write("[CONTENT]\n" + WebContent + "\n")
            st.info('Writing 1 LINE')
            st.warning(WebContent)

else:
    print('Error : could not get articles')

with st.expander("Article"):
    st.code(output)

file.close()

# # Init
# newsapi = NewsApiClient(api_key='3302aaafa5ae4d9eb441f833e249ce77')

# # /v2/top-headlines
# top_headlines = newsapi.get_top_headlines(q='bitcoin',
#                                           sources='bbc-news,the-verge',
#                                           language='en')


# /v2/everything
# all_articles = newsapi.get_everything(q='bitcoin',
#                                       sources='bbc-news,the-verge',
#                                       domains='bbc.co.uk,techcrunch.com',
#                                       from_param='2017-12-01',
#                                       to='2017-12-12',
#                                       language='en',
#                                       sort_by='relevancy',
#                                       page=2)

# /v2/top-headlines/sources
# sources = newsapi.get_sources()



# loader = WebBaseLoader([url])
# data = loader.load()
# content = data[0].page_content.replace('\n', '')
# st.info(content)

# get the content of the artice
# url = 'https://news.google.com/rss/articles/CCAiC3VUSExLWGxTRnFnmAEB?oc=5'
# loader = WebBaseLoader([url])
# data = loader.load()
# WebContent = data[0].page_content.replace('\n', '')
# st.write(WebContent)



