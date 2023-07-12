

import streamlit as st
import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from PIL import Image
from langchain.document_loaders import WebBaseLoader
import requests
from PIL import Image
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from langchain.embeddings import OpenAIEmbeddings

# requirements.txt
# streamlit
# langchain
# requests
# DateTime
# openai
# newsapi
# bs4

st.set_page_config(
    page_title="Text -> Vector",
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
image = Image.open("ironman-banner.jpg")
st.image(image, caption='created by MJ')



url = 'https://www.scmp.com/news/hong-kong/transport/article/3227297/hongkongers-flood-hk-express-website-carrier-gives-away-21626-free-tickets-many-fail-enter-booking'

# Get the current datetime as a string
now = datetime.datetime.now()
Extractionfilename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
sample_text = """
"""


with st.sidebar:
    system_openai_api_key = os.environ.get('OPENAI_API_KEY')
    system_openai_api_key = st.text_input(":key: OpenAI Key :", value=system_openai_api_key)
    os.environ["OPENAI_API_KEY"] = system_openai_api_key


    system_pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    system_pinecone_api_key = st.text_input(":key: PINCONE Key :", value=system_pinecone_api_key)
    os.environ["PINECONE_API_KEY"] = system_pinecone_api_key

    system_pinecone_env_key = os.environ.get('PINECONE_ENV')
    system_pinecone_env_key = st.text_input(":key: PINCONE ENV :", value=system_pinecone_env_key)
    os.environ["PINECONE_ENV"] = system_pinecone_env_key


def read_file(filename):
    try:
        with open(filename, "r") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except:
        print(f"Error: Could not read file '{filename}'.")

def SplitFile():
    with open(Extractionfilename) as f:
        webcontent_file = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.create_documents([webcontent_file])

    st.caption(f' ✔️ Split completed, No. of chunk :  {str(len(chunks))}')

    
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in chunks])
    formatted_total_tokens = "{:,}".format(total_tokens)
    st.caption(f'✔️ Total Tokens: {formatted_total_tokens}')
    st.caption(f'✔️ Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')

def create_vectors():
    embeddings = OpenAIEmbeddings()


def extraction():
    response = requests.get(news_endpoint)

    AllWebLinks = ""
    if response.status_code == 200:
        data = response.json()


        st.caption('Start capture web content:')
        articles = data['articles']
        st.caption(f'Total {len(articles)} web page(s) will  extract')
        NoOfProcess = 1
        with open(Extractionfilename, "w") as file:
            for article in articles:

                if NoOfProcess == 3:
                    break
                 
                title = article['title'] 
                url = article['url']

                webLink = "\n<a href='"  +  article['url']    + "'>" +  article['title']  + "</>"
                AllWebLinks = AllWebLinks + webLink
                
                # get the content of the artice
                loader = WebBaseLoader([url])
                data = loader.load()
                WebContent = data[0].page_content.replace('\n', '')

                if len(WebContent) > 0:
                    WebContent_brief = WebContent[:100]
                    file.write("\n\n")
                    file.write("[TITLE]\n" + title + "\n")
                    file.write("[CONTENT]\n" + WebContent + "\n")
                    st.caption(f' ▶️   processing : ' + str(NoOfProcess) + ' file ' + url) 
                    st.caption(f' ✔️ completed : {WebContent_brief}  ....' )
                    NoOfProcess = NoOfProcess + 1
            if (NoOfProcess > 0):
                st.caption(f'✔️ Extraction completed, total : {str(NoOfProcess - 1)} website.')
                col1 , col2  = st.columns(2)
                with col1:  
                    with st.expander(" Json Format:1"):
                        st.code(data)
                with col2:
                    with st.expander(" Json Format:2"):
                        st.write(data)


    else:
        print('❌❌❌  New API cannot fetch any content')

    with st.expander("Links contained in responses"):
        st.code(AllWebLinks)

    file.close()



def DumpFileContents():
    file_content = read_file(filename=Extractionfilename)
    if file_content is not None:
        with st.expander(f'{Extractionfilename} content'):
            st.code(file_content)
    else:
        print("Error: Could not read file.")


news_endpoint = 'https://newsapi.org/v2/top-headlines?sources=techcrunch&apiKey=3302aaafa5ae4d9eb441f833e249ce77'
st.subheader(f'🌎 :blue[News API Extract]')
st.caption(f'endpoint : {news_endpoint}')
st.caption('Press below Button to start extraction')
if st.button('start', type='primary'):
    extraction()
    DumpFileContents()
    SplitFile()
    create_vectors()
