import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import logging
import uvicorn  # Import uvicorn for the __main__ block

# --- Configuration ---
MODEL_NAME = 'bert-base-uncased'
DIMENSIONS = 3
PERPLEXITY = 5  # Adjust based on expected number of words (must be < n_samples)
GLOBE_RADIUS = 10
SIMILARITY_THRESHOLD = 0.7

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS (Cross-Origin Resource Sharing) ---
origins = [
    "http://localhost:3000",  # Add your frontend origin here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Load Model (Load once on startup) ---
tokenizer = None
model = None

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    logger.info(f"Loading model: {MODEL_NAME}...")
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        model = BertModel.from_pretrained(MODEL_NAME)
        model.eval()  # Set to evaluation mode
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Handle model loading failure appropriately

# --- Helper Function ---
def process_words(words: list[str]):
    if not words:
        return {"words": [], "positions": [], "links": []}
    if not tokenizer or not model:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    logger.info(f"Processing words: {words}")
    # 1. Get Embeddings
    word_vectors = []
    valid_words = []
    for word in words:
        try:
            inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            # Use the [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            word_vectors.append(embedding)
            valid_words.append(word)  # Keep track of words successfully processed
        except Exception as e:
            logger.warning(f"Could not get embedding for word '{word}': {e}")
            continue  # Skip words that cause errors

    if not valid_words:
        return {"words": [], "positions": [], "links": []}

    word_vectors = np.array(word_vectors)

    # Handle case where only one word is valid
    if len(valid_words) == 1:
        # Return a single point at origin
        return {"words": valid_words, "positions": [[0, 0, 0]], "links": []}

    # 2. Dimensionality Reduction (t-SNE)
    # Adjust perplexity if it's >= number of samples
    current_perplexity = min(PERPLEXITY, len(valid_words) - 1)
    if current_perplexity <= 0:  # Need at least 2 samples for perplexity > 0
        logger.warning("Not enough samples for t-SNE. Returning empty positions.")
        return {"words": valid_words, "positions": [], "links": []}

    tsne = TSNE(n_components=DIMENSIONS, random_state=42, perplexity=current_perplexity, n_iter=300, init='pca')
    try:
        word_vectors_3d = tsne.fit_transform(word_vectors)
    except ValueError as e:
        logger.error(f"t-SNE failed: {e}. Word vectors shape: {word_vectors.shape}")
        return {"words": valid_words, "positions": [], "links": []}

    # 3. Map to Sphere
    norms = np.linalg.norm(word_vectors_3d, axis=1, keepdims=True)
    # Avoid division by zero if norm is zero
    norms[norms == 0] = 1e-8  # Replace zero norms with a small number
    word_vectors_sphere = word_vectors_3d / norms * GLOBE_RADIUS

    # 4. Calculate Similarities & Links (based on original embeddings)
    links = []
    if len(valid_words) > 1:
        similarity_matrix = cosine_similarity(word_vectors)
        for i in range(len(valid_words)):
            for j in range(i + 1, len(valid_words)):
                if similarity_matrix[i, j] > SIMILARITY_THRESHOLD:
                    links.append((i, j))

    logger.info(f"Processing complete. Found {len(links)} links.")
    return {
        "words": valid_words,
        "positions": word_vectors_sphere.tolist(),  # Convert numpy array to list for JSON
        "links": links
    }

# --- API Endpoint ---
@app.get("/api/word-data")
async def get_word_data(words_query: str = "dog,cat,puppy,kitten,lion,tiger,wolf,fox,bear,cheetah"):
    """
    Takes a comma-separated string of words, processes them,
    and returns 3D positions and links.
    """
    words_list = [word.strip() for word in words_query.split(',') if word.strip()]

    # Add basic validation
    if not words_list:
        return {"words": [], "positions": [], "links": []}

    try:
        data = process_words(words_list)
        return data
    except Exception as e:
        logger.error(f"Error processing words in endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during word processing: {e}")

if __name__ == "__main__":
    # This block allows running with `python main.py` for simple testing
    uvicorn.run(app, host="0.0.0.0", port=8000) 