from dotenv import load_dotenv
import os
import pymongo
import streamlit as st
import time

from src import vectors_get_embedding_minilm

load_dotenv()
mongo_user = os.getenv("mongo_user")
mongo_pwd = os.getenv("mongo_pwd")
db_name = os.getenv("mongo_db_name")
collection_name = os.getenv("mongo_coll_name")

client = pymongo.MongoClient(f"mongodb+srv://{mongo_user}:{mongo_pwd}@cluster0.sulbktw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client[db_name]
collection = db[collection_name]

st.image(os.path.join("static", "logo.png"))
query = st.text_input(label="What do you fancy?")
header_title = st.header("")
subheader_ingredients = st.subheader("")
subheader_instructions = st.subheader("")

hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
button[title="Link"]{
    visibility: hidden;}
</style>
'''

st.markdown(hide_img_fs, unsafe_allow_html=True)

if query:
    header_title.empty()
    subheader_ingredients.empty()
    subheader_instructions.empty()
    
    query_embedded = vectors_get_embedding_minilm(text=query)
    results = collection.aggregate([
    {
        "$vectorSearch" : {
            "queryVector": query_embedded,
            "path": "title_embedding",
            "numCandidates": 100,
            "limit": 1,
            "index": "title_search"
        }  
    }
    ]).next()
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)
    bar.empty()

    header_title = st.header(results["title"])
    
    subheader_ingredients = st.subheader("INGREDIENTS")
    for ing in eval(results["quantities"]):
        st.markdown(f"- {ing}")
    
    subheader_instructions = st.subheader("INSTRUCTIONS")
    for idx, step in enumerate(eval(results["instructions"])):
        st.markdown(f"**{idx + 1}.** {step}")
