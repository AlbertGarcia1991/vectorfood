"""
This module constains all functions and objects to work with the datasets and 
upload or update the MongoDB service
"""

import os

from dotenv import load_dotenv
import pandas as pd
import pymongo
from tqdm import tqdm

from embeddings import vectors_get_embedding_minilm
from utils import data_raw_path, data_batches_path


def data_post_mongo_db(db_endpoint: str, db_name: str, collection_name: str) -> None:
    """Pushes the recipes with the embeddings to the MondoDB database. This is done reading the batch-processed and
    embedded recipes one by one.

    Args:
        db_endpoint (str): URL of the MongoDB database.
        db_name (str): Name of the MongoDB database.
        collection_name (str): Name of the MongoDB collection inside the above given database name.
    """
    client = pymongo.MongoClient(db_endpoint)
    db = client[db_name]
    collection = db[collection_name]
    print(f"PUSHING AT DB '{db_name}', COLLECTION '{collection_name}'")
    
    for batch_id in tqdm(range(23), desc="UPLOADING BATCHES TO MONGODB"):
        df = pd.read_csv(
            os.path.join(data_batches_path, f"recipes2m_batch_{batch_id}.csv"),
            names=['_id', 'title', 'quantities', 'ingredients', 'instructions', 'title_embedding', 
                   'quantities_embedding', 'quantities_flat_embedding'],
            header=0
        )
        df.loc[:, "_id"] += 1
        recipes_list_dicts = df.to_dict("records")
        for recipe in recipes_list_dicts:
            recipe["title_embedding"] = eval(recipe["title_embedding"])
            recipe["quantities_embedding"] = eval(recipe["quantities_embedding"])
            recipe["quantities_flat_embedding"] = eval(recipe["quantities_flat_embedding"])
        for index in tqdm(range(0, len(recipes_list_dicts), 10000), desc=f"UPLOADING SUB-BATCH"):
            recipes_to_upload = recipes_list_dicts[index:index + 10000]
            collection.insert_many(recipes_to_upload)


def data_process_raw(batch_size: int) -> None:
    """Processes the raw file of recipes2m.csv (containing +2M recipes) by batches, creating the embeddings for the 
    specified columns and storieng each batch as a Pandas DataFrame.
    
    Args:
        batch_size (int): Size of the processed batches.
    """
    for bn in tqdm(range(2231142 // batch_size + 1), desc="MAIN LOOP"):
        if not os.path.isfile(os.path.join(data_batches_path, f"instructions_flat_embedding_{bn}.pkl")):
            start_row = bn * batch_size
            df = pd.read_csv(
                os.path.join(data_raw_path, "recipes_2m.csv"), skiprows=range(1, start_row), nrows=batch_size
            )
            
            with tqdm(total=100, desc=f"BATCH {bn}") as pbar:
                bad_recipes_set = {idx for idx, v in enumerate(df.quantities.to_list()) if len(eval(v)) < 2}
                bad_recipes_set.update({idx for idx, v in enumerate(df.instructions.to_list()) if len(eval(v)) < 2})
                df.drop(bad_recipes_set, axis=0, inplace=True)
                pbar.update(10)

                df["title_embedding"] = vectors_get_embedding_minilm(df.title.values, bs=256)
                pbar.update(10)
                
                df["quantities_embedding"] = vectors_get_embedding_minilm(df.quantities.to_list(), bs=256)
                pbar.update(30)

                df["quantities_flat_embedding"] = vectors_get_embedding_minilm(
                    list(map(lambda x: ";;".join(x), df.quantities.to_list())), bs=256)
                pbar.update(30)
                
                df.to_csv(os.path.join(data_batches_path, f"recipes2m_batch_{bn}.csv"))
                pbar.update(20)


if __name__ == "__main__":   
    data_process_raw(batch_size=50000)

    # ENV variables
    load_dotenv()
    mongo_endpoint = os.getenv("mongo_endpoint")
    mongo_user = os.getenv("mongo_user")
    mongo_pwd = os.getenv("mongo_pwd")
    mongo_db_name = os.getenv("mongo_db_name")
    mongo_coll_name = os.getenv("mongo_coll_name")
    mongo_endpoint = mongo_endpoint.replace("$USER", mongo_user)
    mongo_endpoint = mongo_endpoint.replace("$PWD", mongo_pwd)
    data_post_mongo_db(db_endpoint=mongo_endpoint, db_name=mongo_db_name, collection_name=mongo_coll_name)
