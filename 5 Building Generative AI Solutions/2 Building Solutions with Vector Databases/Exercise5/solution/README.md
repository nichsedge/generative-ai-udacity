# Multimodal Search

In this final exercise, we will learn how to use vector databases to search through images using natural language. 

We will be searching through an open source image dataset using an open source model called CLIP.
This model is able to encode both images and text into the same embedding space, allowing us to retrieve images that are similar to a user question.


## Setup CLIP model

First, let's prepare the [CLIP](https://huggingface.co/docs/transformers/model_doc/clip) model to encode the images.
We want to setup two things:
1. a model to encode the image
2. a processor to prepare the image to be encoded

Fill in the code to initialize a pre-trained model and processor.

**SOLUTION** By reading the CLIP documentation, we see that we can use the `from_pretrained` classmethods to initialize the model and the processor


## Setup data model

The dataset itself has an image field and an integer label.
We'll also need an embedding vector (CLIP produces 512D vectors) field.

For this problem, please a field named "vector" to the Image class below
that is a 512D vector.

The image that comes out of the raw dataset is a PIL image. So we'll add
some conversion code between PIL and bytes to make it easier for serde.


## Image processing function

Next we will implement a function to process batches of data from the dataset.
We will be using the `zh-plus/tiny-imagenet` dataset from huggingface datasets.
This dataset has an `image` and a `label` column.

For this problem, please fill in the code to extract the image embeddings from
the image using the CLIP model.

**SOLUTION** Here we'll use the `get_image_features` method on the model instance


## Table creation

Please create a LanceDB table called `image_search` to store the image, label, and vector.

**SOLUTION** this should be familiar to you by now. Let's call `lancedb.connect` then
`db.create_table`

## Adding data

Now we're ready to process the images and generate embeddings.
Please write a function called `datagen` that calls `process_image` on each image in the validation set (10K images) and return a list of Image instances.

**HINT**
1. You may find it faster to use the [dataset.map](https://huggingface.co/docs/datasets/process#map) function.
2. You'll want to store the `image_bytes` field that is returned by `process_image`.

**SOLUTION**
1. We'll call `load_dataset` and retrieve the `valid` split
2. We then map `process_image` and construct Image instances


Now call the function you just wrote and add the generated instances to the LanceDB table


## Encoding user queries

We have image embeddings, but how do we generate the embeddings for the user query?
Furthermore, how can we possibly have the same features between the image embeddings
and text embeddings. This is where the power of CLIP comes in.

Please a function to turn user query text into an embedding
in the same latent space as the images. 

**HINT** 
You can refer to the [CLIPModel documention](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPModel)



## Core search function

Now let's write the core search function `find_images`, that takes a text query as input, and returns a list of PIL images that's most similar to the query.

**SOLUTION**
First, we need to call `embed_func` on the query to generate the text embedding.
Then, we'll search through the LanceDB to get images most similar to the query.
And we'll convert the resrults into Image instances.
Finally, we'll call the `Image.to_pil` method to convert the bytes into PIL images




## Create an App

Let's use gradio to create a small app to search through the images.
Please fill in the code below:
1. Create a [text input](https://www.gradio.app/docs/textbox) where the user can type in a query
2. Create a "Submit" [button](https://www.gradio.app/docs/button) that finds similar images to the input query and display the resulting images
3. A [Gallery component](https://www.gradio.app/docs/gallery) that displays the images

**SOLUTION**

1. The input is just `gr.Textbox` and you'll want to assign that to some variable
2. The submit button is `gr.Button`. Again, assign it to some variable
3. The gallery is `gr.Gallery`
4. You'll want to define the on click behavior for the button such that it uses
   the textbox value as input, calls the `find_images` function you wrote above,
   and uses the gallery to display the results


## Summary

Congrats! 

Through this exercise, you learned how to use CLIP to generate image and text embeddings. You've mastered how to use vector databases to enable searching through images using natural language. And you even created a simple app to show off your work. 

Great job!


```python

```
