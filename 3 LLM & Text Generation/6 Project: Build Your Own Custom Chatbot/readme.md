## Project: Build Your Own Custom Chatbot

In this project you will change the dataset to build your own custom chatbot. You will build the chatbot manually using just basic packages like `openai` and `pandas`, not using frameworks like LangChain, in order to gain a deeper understanding of how these systems work "under the hood".

### What Will You Build?

When you have completed this project, you will have a custom OpenAI chatbot using a scenario of your choice. You will be responsible for selecting a data source, explaining why the data source is appropriate for the task, incorporating the data source into the custom chatbot code, and writing questions to demonstrate the performance of the custom prompt.

### Data Sources

There are two main data sources we recommend using for this project: Wikipedia API and CSV data.

The Wikipedia API will be most similar to the examples shown in the demos and exercises you have previously seen. You can use any article other than [2022(opens in a new tab)](https://en.wikipedia.org/wiki/2022) or [2023 Syria-Turkey Earthquake(opens in a new tab)](https://en.wikipedia.org/wiki/2023_Turkey%E2%80%93Syria_earthquake) as long as it fulfills the requirements.

We have also provided a `data` directory containing CSV files that you can use for the project. These are:

- `2023_fashion_trends.csv` - this file contains reports and quotes about fashion trends for 2023. Each row includes the source URL, article title, and text snippet.
- `character_descriptions.csv` - this file contains character descriptions from theater, television, and film productions. Each row contains the name, description, medium, and setting. All characters were invented by an OpenAI model.
- `nyc_food_scrap_drop_off_sites.csv` - this file contains locations, hours, and other information about food scrap drop-off sites in New York City. This information was retrieved in early 2023, and you can also get the latest version from [this open data portal(opens in a new tab)](https://dev.socrata.com/foundry/data.cityofnewyork.us/if26-z6xq).

You may also source your own data. For example, you might want to use web scraping or other documents you have on hand. The dataset must have **at least 20 rows**, and it must be composed of **text data**. OpenAI language models are not optimized for numeric or logical reasoning, so number-heavy data like budgets, sensor data, or inventory are not appropriate.

### Custom Scenario

In addition to the technical component of preparing and incorporating a new dataset, you need to explain why this dataset is appropriate for the task. **If the model responds the same way regardless of whether custom data is provided, that means that the dataset was not appropriate for the task.** You will explain your dataset choice in two ways:

1. First, at the start of the notebook, you will write a short paragraph describing your dataset choice and setting up the scenario of when this customization would be useful.
2. Second, at the end of the notebook, you will demonstrate the model Q&A before and after the customization has been performed in order to highlight the changes.

## Project Instructions

Work for this project will primarily be completed in the Jupyter Notebook titled `project.ipynb`. This notebook contains `TODO` indicators in the four places where you must complete tasks. These four tasks correspond to the four rubric items.

### 1. Choose a Dataset and Explain the Scenario

Consider the dataset choices described on the Project Overview page and select which dataset you are going to use for this project.

In the second cell of the notebook, there is a `TODO` in a Markdown cell. Double-click on this text to edit it. Write a paragraph explaining which dataset you are using and why.

### 2. Prepare the Dataset for the Custom Query Process

None of the dataset options are in exactly the right format for this process. You need to get them into a format so they can be loaded as a `pandas` dataframe with a column called `"text"`.

Note that you are not required to use `pandas` to manipulate the data. You can use Excel/Google Sheets, a text editor, or whatever other software you are comfortable using. If you are using other software to reshape or clean the data, make sure you upload the data file to the `data` folder then use `pandas` to load it.

### 3. Perform the Custom Query Process

Integrate your dataset with the custom query code that was provided previously. You can copy and paste as needed. Just make sure that you are using your custom dataset and not one of the datasets used in the course content!

### 4. Write Questions to Demonstrate Custom Performance

In the last cells of the notebook, there are spaces for "Question 1" and "Question 2". Write at least two questions that show how the model answers differently with and without your custom prompt.

# Project: Build Your Own Custom Chatbot

## Code

| Criteria                | Submission Requirements                                                                                                                                    |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data Wrangling          | Project dataset is loaded into a `pandas` dataframe containing at least 20 rows. Each row in the dataset contains a text sample in a column named `"text"` |
| Custom Query Completion | The project successfully sends a custom query with information from the project dataset to the OpenAI model and gets a response                            |
|                         |                                                                                                                                                            |

## Scenario

|Criteria|Submission Requirements|
|---|---|
|Demonstrate that custom prompts enhance the performance of the OpenAI model|The notebook includes at least two questions that result in different answers when using the custom prompt compared to a basic prompt to the same model.<br><br>This means that there should be 4 total Q&A pairs: 2 questions with the custom answers and those same 2 questions with the basic answers|
|Select a dataset for generating custom prompts and explain the rationale|The notebook includes at least one sentence explaining why the chosen dataset is appropriate for this application|

### Suggestions to Make Your Project Stand Out

1. Use a `while` loop and the built-in `input` function to allow the user to type questions repeatedly rather than having to edit the content of strings in the code cells
2. Take a closer look at your dataset and perform additional data cleaning to remove irrelevant content or augment the content