# VectorFood

## Description
The current repository hosts a personal project developed with the goal of learning more about LLMs and sentence embeddings. 
I have embedded the title, inrgedients, and insturctions of more than 2 million culinary recipes (from database [Recipes2M](https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m)) and store them into a MongoDB. Then, created a index search to find recipes by embedding similarity a given query. All this has been condensed into a Streamlit app available online.

## Usage
Navigate to the website [VectorFoodApp](https://vectorfood.streamlit.app) to try it out.

## Features
* Recipes titles, ingredients, and instructions embedded using HuggingFace's [MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) sentence transformer.
* Streamlit App with search bar and recipe display
* Similarity search made by embedding the user query and returning the closest match across all embedded title.

## Versions
* **0.1**: Basic version with a single input bar which searched for the closes embedded title to the query 

## To Do
* Create and upload ingredients DB with nutritional and organoleptic properties
* Make both ingredients and recipes DB available to explore on different pages of the App
* Expand searching options adding:
  *  Dietary constraints (e.g. vegetarian, vegan, allergens, ...)
  * Culinary origin (e.g. Italian, Spanish, Thai, ...)
  * Ingredients must appear on the recipe (given them explicitly)
  * Ingredients must not appear on the recipe (given them explicitly)
 * Ingredients similarity and dissimilarity (embedding) 
 * Ingredients replacement
 * GenAI to generate image of the recipe
