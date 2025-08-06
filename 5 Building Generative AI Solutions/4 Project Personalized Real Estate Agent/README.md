# HomeMatch Application

## Overview
"HomeMatch" is an innovative application that leverages Large Language Models (LLMs) and vector databases to provide a personalized real estate listing experience. It interprets buyer preferences in natural language, semantically searches for matching properties, and then rewrites listing descriptions to highlight aspects most relevant to the buyer.

## Features
- **LLM-powered Listing Generation:** Generates realistic real estate listings.
- **Vector Database Integration:** Stores property listings as vector embeddings for efficient semantic search.
- **Personalized Search:** Matches buyer preferences with relevant properties using semantic search.
- **Dynamic Listing Personalization:** Rewrites listing descriptions to resonate with individual buyer needs without altering factual information.

## How to Run

### Prerequisites
- Python 3.11 or higher
- `uv` for Python package management (recommended)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Install dependencies using `uv`:**
    ```bash
    uv sync
    ```
3.  **Set up OpenAI API keys:**
    Create a `.env` file in the project root directory and add your OpenAI API key and base URL:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    OPENAI_API_BASE="https://openai.vocareum.com/v1"
    ```
    Replace `"your_openai_api_key_here"` with your actual OpenAI API key.

### Running the Application
To run the HomeMatch application, execute the `HomeMatch.py` script:

```bash
python HomeMatch.py
```

The script will:
1.  Generate 10 real estate listings and save them to `listings.txt`.
2.  Store these listings in a ChromaDB vector database (persisted in `./chroma_db`).
3.  Use a predefined set of buyer preferences to perform a semantic search.
4.  Retrieve the top 3 matching listings.
5.  Personalize the descriptions of these retrieved listings based on the buyer's preferences and print them to the console.

## Project Structure
- `HomeMatch.py`: The main application script containing all the logic for listing generation, vector database interaction, search, and personalization.
- `listings.txt`: A file containing the synthetically generated real estate listings.
- `chroma_db/`: Directory where the ChromaDB vector database is persisted.
- `pyproject.toml`: Project dependencies and metadata.
- `.env`: (Not committed) Used to store environment variables like API keys.

## Testing
You can modify the `questions` and `answers` variables in the `HomeMatch.py` script to simulate different buyer preferences and observe how the personalized listings change.

## Deliverables
- `HomeMatch.py` (or `HomeMatch.ipynb` if preferred)
- `listings.txt` (generated real estate listings)
- `README.md` (this documentation)
- `chroma_db/` (persisted vector database)