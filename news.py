

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
import pinecone
from langchain.vectorstores import Pinecone

# Answering in Natural Language using an LLM
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI




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

#################
# Global Variables
################# 
index_name = 'idx1'
url = 'https://www.scmp.com/news/hong-kong/transport/article/3227297/hongkongers-flood-hk-express-website-carrier-gives-away-21626-free-tickets-many-fail-enter-booking'
data = None


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

    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))

    # delete all index
    # indexes = pinecone.list_indexes()    
    # for i in indexes:
    #     pinecone.delete_index(i)
    #     st.caption(f' ✔️ Pinecone - delete index')

    # # create index     
    # if index_name not in pinecone.list_indexes():
    #     pinecone.create_index(index_name, dimension=1536, metric='cosine')
    #     st.caption(f'✔️ Pinecone - create index {index_name} ')

    # indexes = pinecone.list_indexes()    
    # for i in indexes:
    #     st.caption(f' ✔️ Pinecone - index : {i.name} ')
    now = datetime.datetime.now()
    st.caption(f'✔️ {now.strftime("%H-%M-%S")} : Start create vector store for document {Extractionfilename}')
    vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    now = datetime.datetime.now()
    st.caption(f'✔️ {now.strftime("%H-%M-%S")} : vector store created on index :{index_name}')
    
    query = 'anything about ebay ?'

    st.subheader('1. LLM searach - new chunks')
    st.caption(f'💬 Start Similarity Search (VectorStore): {query} ')
    result = vector_store.similarity_search(query)
    st.caption(f'🟢 Result  : {result} ')
    found_result = [document.page_content for document in result]
    st.caption(f'🟢 Extracted  : {found_result} ')


    ### Answering in Natural Language using an LLM
    st.subheader('2. LLM searach by Chain')
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    st.caption(f'💬 Start Similarity Search (LLM): {query} ')
    now = datetime.datetime.now()
    st.caption(f'✔️ {now.strftime("%H-%M-%S")} : Start LLM Similarity')
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    LLM_Answer = chain.run(query)
    now = datetime.datetime.now()
    st.caption(f'🟢  {now.strftime("%H-%M-%S")} Completed  : {LLM_Answer} ')


    ####  already have an index, you can load it like this   
    st.subheader('Vector searach - index')
    now = datetime.datetime.now()
    st.caption(f'✔️ {now.strftime("%H-%M-%S")} : Start search by existing Index : {index_name}')
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    docs = docsearch.similarity_search(query)
    existing_index_found_result = [document.page_content for document in docs]
    now = datetime.datetime.now()
    st.caption(f'🟢  {now.strftime("%H-%M-%S")} Completed   : {existing_index_found_result} ')

    




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

                if NoOfProcess == 2:
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
                with st.expander(" Json Format:1"):
                    st.code(data)


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
