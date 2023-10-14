import os
from typing import List

import chromadb
import numpy as np

from slashgpt.dbs.db_base import VectorDBBase
from slashgpt.dbs.vector_engine import VectorEngine


class DBChroma(VectorDBBase):
    def __init__(self, embeddings: dict, vector_engine: VectorEngine, verbose: bool):
        super().__init__(embeddings, vector_engine, verbose)
        db_path = embeddings.get("db_path") if embeddings.get("db_path") else os.path.normpath(os.path.expanduser("~/.slashgpt/chroma-db"))
        table_name = embeddings.get("name")

        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_collection(table_name)

    def fetch_data(self, query_embedding: List[float]) -> List[str]:
        res = self.collection.query(
            query_embeddings=[np.array(query_embedding).tolist()],
            n_results=5,
        )
        return list(map(lambda x: "".join(x), res["documents"]))
