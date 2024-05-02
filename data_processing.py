"""
This module constains all functions and objects to work with the datasets and 
upload or update the MongoDB service
"""

import os
import pickle
from typing import Set

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pymongo
from tqdm import tqdm

from embeddings import vectors_get_embedding_minilm
from utils import data_raw_path, data_batches_path, data_embedded_path


def data_post_mongo_db(db_endpoint: str, db_name: str, collection_name: str) -> None:
    with open(os.path.join(data_embedded_path, "recipes_embedded.pkl"), "rb") as f:
        data_embedded = pickle.load(f)["recipes_embedded"]
    client = pymongo.MongoClient(db_endpoint)
    db = client[db_name]
    collection = db[collection_name]
    collection.insert_many(data_embedded)


def data_process_batches(keys_processes: Set[int] = None) -> int:
    # TODO: load pkl files, join together all batches, generate dict with expected format, add _id to each element of the list
    # recipes_list_to_db = []
    # for idx, row in df.iterrows():           
    #     recipes_list_to_db.append(
    #         {
    #             "_id": idx + 1,
    #             "title": row.title,
    #             "quantities": eval(row.quantities),
    #             "ingredients": eval(row.ingredients),
    #             "instructions": eval(row.instructions),
    #             "title_emdedding": row.title_emdedding
    #         }
    #     )
    # filename = f"batch_{offset_batch + pid}.pkl"
    # with open(os.path.join(data_batches_path, filename), "wb") as f:
    #     pickle.dump(recipes_list_to_db, f)
    # TODO: Before we were saving a dict with "idx" key, now is the lsit of values to update the db directly
    with open(os.path.join(data_embedded_path, "recipes_embedded.pkl"), "wb") as f:
        pickle.dump(out_dict, f)
        
    return len(keys_processes)


def data_process_raw(df: pd.DataFrame, batch_number: int) -> None:
    with tqdm(total=100, desc=f"BATCH {batch_number}") as pbar:
        title_emdedding = vectors_get_embedding_minilm(df.title.values, bs=256)
        with open(os.path.join(data_batches_path, f"title_embedding_{batch_number}.pkl"), "wb") as f:
            pickle.dump(title_emdedding, f)
        pbar.update(10)
        
        bad_recipes_set = {idx for idx, v in enumerate(df.quantities.to_list()) if len(eval(v)) < 2}
        bad_recipes_set.update({idx for idx, v in enumerate(df.instructions.to_list()) if len(eval(v)) < 2})
        quantities_processed = [eval(v) for idx, v in enumerate(df.quantities.to_list()) if idx not in bad_recipes_set]
        instructions_processed = [eval(v) for idx, v in enumerate(df.instructions.to_list()) if idx not in bad_recipes_set]
        pbar.update(20)
        
        quantities_emdedding = vectors_get_embedding_minilm(quantities_processed, bs=256)
        with open(os.path.join(data_batches_path, f"quantities_embedding_{batch_number}.pkl"), "wb") as f:
            pickle.dump(quantities_emdedding, f)
        pbar.update(40)

        quantities_flat_embedding = vectors_get_embedding_minilm(list(map(lambda x: ";;".join(x), quantities_processed)), bs=256)
        with open(os.path.join(data_batches_path, f"quantities_flat_embedding_{batch_number}.pkl"), "wb") as f:
            pickle.dump(quantities_flat_embedding, f)
        pbar.update(60)
            
        instructions_embedding = vectors_get_embedding_minilm(instructions_processed, bs=256)
        with open(os.path.join(data_batches_path, f"instructions_embedding_{batch_number}.pkl"), "wb") as f:
            pickle.dump(instructions_embedding, f)
        pbar.update(80)
        
        instructions_flat_embedding = vectors_get_embedding_minilm(list(map(lambda x: ";;".join(x), instructions_processed)), bs=256)
        with open(os.path.join(data_batches_path, f"instructions_flat_embedding_{batch_number}.pkl"), "wb") as f:
            pickle.dump(instructions_flat_embedding, f)
        pbar.update(100)


if __name__ == "__main__":
    batch_size = 100000
    for bn in tqdm(range(2231142 // batch_size + 1), desc="MAIN LOOP"):
        if not os.path.isfile(os.path.join(data_batches_path, f"instructions_flat_embedding_{bn}.pkl")):
            start_row = bn * batch_size
            df = pd.read_csv(os.path.join(data_raw_path, "recipes_2m.csv"), skiprows=range(1, start_row), nrows=batch_size)
            data_process_raw(df=df, batch_number=bn)
    
    # ENV variables
    # load_dotenv()
    # mongo_endpoint = os.getenv("mongo_endpoint")
    # mongo_user = os.getenv("mongo_user")
    # mongo_pwd = os.getenv("mongo_pwd")
    # mongo_db_name = os.getenv("mongo_db_name")
    # mongo_coll_name = os.getenv("mongo_coll_name")
    # mongo_endpoint = mongo_endpoint.replace("$USER", mongo_user)
    # mongo_endpoint = mongo_endpoint.replace("$PWD", mongo_pwd)
    
    # n_processed_recipes = data_process_batches()
    # print(f"PUSHING {n_processed_recipes} recipes embedded")
    # data_post_mongo_db(db_endpoint=mongo_endpoint, db_name=mongo_db_name, collection_name=mongo_coll_name)
