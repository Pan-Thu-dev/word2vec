# Word2Vec Visualization

This project visualizes word embeddings from a BERT model in a 3D interactive globe. Words that are semantically related appear closer together on the globe, and highly similar words are connected by lines.

## Project Structure

- `backend/`: Python FastAPI backend that provides word embeddings using the BERT model
- `frontend/`: Next.js frontend that visualizes the word embeddings in a 3D globe using Three.js

## Prerequisites

- **Python 3.8+**: Install from [python.org](https://python.org/)
- **Node.js and npm**: Install from [nodejs.org](https://nodejs.org/)
- **Redis**: Install from [redis.io](https://redis.io/download)
- **Git**: (Optional but recommended) for version control

## Setup

### Redis Setup

1. Install Redis for your platform (see [REDIS_SETUP.md](REDIS_SETUP.md) for detailed instructions)
2. Start the Redis server:
   ```bash
   # On Windows
   redis-server

   # On macOS
   brew services start redis

   # On Linux
   sudo systemctl start redis-server
   ```

### Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables (optional):
   - Create a `.env` file in the backend directory
   - Add your Redis connection details:
     ```
     REDIS_URL=redis://localhost:6379
     REDIS_PASSWORD=
     REDIS_DB=0
     MODEL_NAME=bert-base-uncased
     ```

5. Start the backend server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Frontend

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Usage

1. Enter comma-separated words in the input field.
2. Click the "Add Words" button to add them to the visualization.
3. The words will be displayed on a 3D globe, with related words positioned closer together.
4. Use your mouse to rotate, zoom, and pan the globe to explore the word relationships.
5. Click "Reset All" to clear all words and start over.

## How It Works

1. The frontend allows you to incrementally add words to the visualization.
2. Words are stored in Redis for persistence across server restarts.
3. The backend uses a BERT model to generate embeddings for each word.
4. These high-dimensional embeddings are reduced to 3D using t-SNE.
5. The 3D coordinates are projected onto a sphere (the globe).
6. Words with high similarity (based on cosine similarity) are connected by lines.
7. The frontend renders this data as an interactive 3D visualization.

## Advanced Configuration

For advanced Redis configuration options and troubleshooting, see [REDIS_SETUP.md](REDIS_SETUP.md). 