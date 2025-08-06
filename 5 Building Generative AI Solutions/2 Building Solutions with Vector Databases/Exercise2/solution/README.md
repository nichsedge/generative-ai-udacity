# Embedding functions

Using the appropriate model, anything can be embedded.
Different models result in different latent spaces and different embedding dimensions. Things that are similar in one embedding model can be very far from each other in the latent space created by another embedding model.

## Turning text into vectors

For this exercise we're going to use embeddings to do semantic search Rick and Morty quotes

### Installation

Installing Sentence Transformer can be done via `pip install sentence-transformers`

### Write function to generate embeddings from text

Write a function that turns text into embeddings using Sentence Transformers.

**HINT**
1. Choose a [pre-trained model](https://www.sbert.net/docs/pretrained_models.html), you don't need to create your own
2. See the API documentation and examples for Sentence Transformers to see how to encode text

**SOLUTION**
Sentence Transformers make this pretty easy. First we load the model using the model name you chose.
Then we call the `model.encode()` function to generate the embeddings. If you pass in a single string,
then a 1D numpy array is returned. Otherwise a 2D array is returned.

Additional exercises:

Q: How many dimensions is each embedding?
A: This depends on the model you passed in. For `paraphrase-MiniLM-L6-v2`, for example, it's 384

Q: Are the embeddings normalized already?
A: No they're not. We may want to add a step in the embedding function to normalize all vectors. OR we use cosine instead of Euclidean distance

### Read data

Now let's read the quotes from the included text file (source: https://parade.com/tv/rick-and-morty-quotes). 

**SOLUTION** Here each quote lives on it's own line of text. So we just do `readlines` here. Remember to use the context manager to close resources. 

### Let's put it together

Now let's build a retrieval example together using the embeddings you just generated.

Vector search is useful for retrieving data that's not part of the model's training data.

For example, if we asked the following question to ChatGPT, we get some generic sounding answer wrapping around the core "she does not make any statements about causing her parents misery".

But if we're able to use the specific quotes we have, we would get a different answer. This usually involves two steps:

1. Retrieving the quotes that are most relevant to the question.
2. Instruct ChatGPT to answer based on those quotes instead.