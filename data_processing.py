from dotenv import load_dotenv
import os
import pickle
import pymongo
from typing import Set

from utils import data_batches_path, data_embedded_path


def data_post_MogoDB(db_name: str, collection_name: str) -> None:
    with open(os.path.join(data_embedded_path, "recipes_embedded.pkl"), "rb") as f:
        data_embedded = pickle.load(f)["recipes_embedded"]
        
    client = pymongo.MongoClient(f"mongodb+srv://{mongo_user}:{mongo_pwd}@cluster0.sulbktw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client[db_name]
    collection = db[collection_name]
    collection.insert_many(data_embedded)


def data_process_batches(keys_processes: Set[int] = None) -> int:
    batch_files = [f for f in os.listdir(data_batches_path) if f.endswith(".pkl")]
    
    if not keys_processes:
        keys_processes = set()

    recipes_embedded = []
    for file in batch_files:
        with open(os.path.join(data_batches_path, file), "rb") as f:
            data = pickle.load(f)
        for recipe in data:
            id = recipe["_id"]
            if id not in keys_processes:
                keys_processes.add(id)
                recipes_embedded.append(recipe)
    
    out_dict = {
        "idx": keys_processes,
        "recipes_embedded": recipes_embedded
    }
    with open(os.path.join(data_embedded_path, "recipes_embedded.pkl"), "wb") as f:
        pickle.dump(out_dict, f)
        
    return len(keys_processes)


if __name__ == "__main__":
    # ENV variables
    load_dotenv()
    mongo_user = os.getenv("mongo_user")
    mongo_pwd = os.getenv("mongo_pwd")
    mongo_db_name = os.getenv("mongo_db_name")
    mongo_coll_name = os.getenv("mongo_coll_name")
    
    n_processed_recipes = data_process_batches()
    print(f"PUSHING {n_processed_recipes} recipes embedded")
    data_post_MogoDB(db_name=mongo_db_name, collection_name=mongo_coll_name)
