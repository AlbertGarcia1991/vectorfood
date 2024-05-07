"""
Streamlit app
"""
import os
import time
from typing import List, Union

import pymongo
from sentence_transformers import SentenceTransformer
import streamlit as st


# HF Embedding
def vectors_get_embedding_minilm(text: Union[List[str], str]) -> List[float]:
    """
    Get the array embedding of the given input using the Huggineface's all-MiniLM-L6-v2
    sentence transformer model.

    Args:
        text (Union[List[str], str]): Text or list of texts to embed.

    Returns:
        List[float]: List containing the 284 embedded float values.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(text).tolist()


# Connection to MongoDB
client = pymongo.MongoClient(st.secrets["mongo"]["connection_string"])
db = client["vectorfood"]
collection = db["recipesdb"]


# Header Image
st.image(os.path.join("static", "logo.png"))
query = st.text_input(label="What do you fancy?")
header_title = st.header("")
subheader_ingredients = st.subheader("")
subheader_instructions = st.subheader("")
HIDE_IMG_HTML = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
button[title="Link"]{
    visibility: hidden;}
</style>
'''
st.markdown(HIDE_IMG_HTML, unsafe_allow_html=True)  # Hide header image's expand button

if query:
    header_title.empty()
    subheader_ingredients.empty()
    subheader_instructions.empty()
    
    query_embedded = vectors_get_embedding_minilm(text=query.lower())
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
    # latest_iteration = st.empty()
    # bar = st.progress(0)
    # for i in range(100):
    #     bar.progress(i + 1)
    #     time.sleep(0.01)
    # bar.empty()

    header_title = st.header(results["title"])
    
    subheader_ingredients = st.subheader("INGREDIENTS")
    for ing in eval(results["quantities"]):
        st.markdown(f"- {ing}")
    
    subheader_instructions = st.subheader("INSTRUCTIONS")
    for idx, step in enumerate(eval(results["instructions"])):
        st.markdown(f"**{idx + 1}.** {step}")
