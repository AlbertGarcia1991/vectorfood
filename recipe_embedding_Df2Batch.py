import multiprocessing
import os
import pandas as pd
import pickle
import time
from tqdm import tqdm

from embeddings import vectors_get_embedding_minilm
from utils import data_raw_path, data_batches_path


# TODO: Use Dask or PySpark

def data_process_raw(df: pd.DataFrame, pid: int, offset_batch: int) -> None:
    if pid == N_PROCESSES - 1:
        tqdm.pandas()
        df["title_emdedding"] = df["title"].progress_map(lambda x: vectors_get_embedding_minilm(x))
    else:
        df["title_emdedding"] = df["title"].map(lambda x: vectors_get_embedding_minilm(x))
    
    df_dict = []
    for idx, row in df.iterrows():           
        df_dict.append(
            {
                "_id": idx + 1,
                "title": row.title,
                "quantities": eval(row.quantities),
                "ingredients": eval(row.ingredients),
                "instructions": eval(row.instructions),
                "title_emdedding": row.title_emdedding
            }
        )
    filename = f"batch_{offset_batch + pid}.pkl"
    with open(os.path.join(data_batches_path, filename), "wb") as f:
        pickle.dump(df_dict, f)
    print(f"FINISHED PROCESS {pid}")


def data_main() -> None:
    df_raw = pd.read_csv(os.path.join(data_raw_path, "recipes_2m.csv"), index_col=0)
    
    processes = []
    for i in range(N_PROCESSES):
        start_idx = OFFSET + i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        p = multiprocessing.Process(target=data_process_raw, args=(df_raw.loc[start_idx:end_idx, :], i, OFFSET_BATCH))
        processes.append(p)
    start_time = time.time()
    for idx, p in enumerate(processes):
        print(f"STARTING PROCESS {idx}")
        p.start()
    for p in processes:
        p.join()
    print("FINISHED: ", time.time() - start_time)
 
 
if __name__ == "__main__":
    N_PROCESSES = 10
    OFFSET = 12000
    BATCH_SIZE = 100000
    OFFSET_BATCH = 12

    data_main()
