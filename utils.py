from sentence_transformers import SentenceTransformer
import os
model_path = os.path.join(os.path.dirname(__file__), "models", "all-mpnet-base-v2")
embedding_model = SentenceTransformer(model_path)