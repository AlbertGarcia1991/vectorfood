"""
Streamlit app
"""
import os
from typing import List, Union

import pymongo
from sentence_transformers import SentenceTransformer
import streamlit as st


st.set_page_config(page_title="VectorFood", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)


# HF Embedding
def vectors_get_embedding_minilm(text: Union[List[str], str]) -> List[float]:
    """
    Get the array embedding of the given input using the Huggineface's all-MiniLM-L6-v2
    sentence transformer model.

    Args:
        text (Union[List[str], str]): Text or list of texts to embed.

    Returns:
        List[float]: List containing the 384 embedded float values.
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


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: orange;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://github.com/AlbertGarcia1991/vectorfood" target="_blank">Albert Garcia Plaza</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

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

    header_title = st.header(results["title"])
    
    subheader_ingredients = st.subheader("INGREDIENTS")
    for ing in eval(results["quantities"]):
        st.markdown(f"- {ing}")
    
    subheader_instructions = st.subheader("INSTRUCTIONS")
    for idx, step in enumerate(eval(results["instructions"])):
        st.markdown(f"**{idx + 1}.** {step}")
