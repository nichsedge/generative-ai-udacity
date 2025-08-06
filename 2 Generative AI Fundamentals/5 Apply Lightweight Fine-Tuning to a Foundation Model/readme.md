## Project Introduction

In this project, you will apply **parameter-efficient fine-tuning** using the Hugging Face `peft` library.

## Project Summary

In this project, you will bring together all of the essential components of a PyTorch + Hugging Face training and inference process. Specifically, you will:

1. Load a pre-trained model and evaluate its performance
2. Perform parameter-efficient fine tuning using the pre-trained model
3. Perform inference using the fine-tuned model and compare its performance to the original model

## Key Concepts

Hugging Face PEFT allows you to fine-tune a model without having to fine-tune all of its parameters.

Training a model using Hugging Face PEFT requires two additional steps beyond traditional fine-tuning:

1. Creating a **PEFT config**
2. **Converting the model into a** **PEFT model** using the PEFT config

Inference using a PEFT model is almost identical to inference using a non-PEFT model. The only difference is that it must be **loaded as a PEFT model**.

## Training with PEFT

### Creating a PEFT Config

The PEFT config specifies the adapter configuration for your parameter-efficient fine-tuning process. The base class for this is a `PeftConfig`, but this example will use a `LoraConfig`, the subclass used for low rank adaptation (LoRA).

A LoRA config can be instantiated like this:

`from peft import LoraConfig config = LoraConfig()`

Look at the LoRA adapter documentation for additional hyperparameters that can be specified by passing arguments to `LoraConfig()`. [Hugging Face LoRA conceptual guide(opens in a new tab)](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) also contains additional explanations.

### Converting a Transformers Model into a PEFT Model

Once you have a PEFT config object, you can load a Hugging Face `transformers` model as a PEFT model by first loading the pre-trained model as usual (here we load GPT-2):

`from transformers import AutoModelForCausalLM model = AutoModelForCausalLM.from_pretrained("gpt2")`

Then using `get_peft_model()` to get a trainable PEFT model (using the LoRA config instantiated previously):

`from peft import get_peft_model lora_model = get_peft_model(model, config)`

### Training with a PEFT Model

After calling `get_peft_model()`, you can then use the resulting `lora_model` in a training process of your choice (PyTorch training loop or Hugging Face `Trainer`).

### Checking Trainable Parameters of a PEFT Model

A helpful way to check the number of trainable parameters with the current config is the `print_trainable_parameters()` method:

`lora_model.print_trainable_parameters()`

Which prints an output like this:

`trainable params: 294,912 || all params: 124,734,720 || trainable%: 0.23643136409814364`

### Saving a Trained PEFT Model

Once a PEFT model has been trained, the standard Hugging Face `save_pretrained()` method can be used to save the weights locally. For example:

`lora_model.save_pretrained("gpt-lora")`

Note that this **only saves the adapter weights** and not the weights of the original Transformers model. Thus the size of the files created will be much smaller than you might expect.

## Inference with PEFT

### Loading a Saved PEFT Model

Because you have only saved the adapter weights and not the full model weights, you can't use `from_pretrained()` with the regular Transformers class (e.g., `AutoModelForCausalLM`). Instead, you need to use the PEFT version (e.g., `AutoPeftModelForCausalLM`). For example:

`from peft import AutoPeftModelForCausalLM lora_model = AutoPeftModelForCausalLM.from_pretrained("gpt-lora")`

After completing this step, you can proceed to use the model for inference.

### Generating Text from a PEFT Model

You may see examples from regular Transformer models where the input IDs are passed in as a positional argument (e.g., `model.generate(input_ids)`). For a PEFT model, they must be passed in as a keyword argument (e.g., `model.generate(input_ids=input_ids)`). For example:

`from transformers import AutoTokenizer tokenizer = AutoTokenizer.from_pretrained("gpt2") inputs = tokenizer("Hello, my name is ", return_tensors="pt") outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10) print(tokenizer.batch_decode(outputs))`

## Documentation Links

- [Hugging Face PEFT configuration(opens in a new tab)](https://huggingface.co/docs/peft/package_reference/config)
- [Hugging Face LoRA adapter(opens in a new tab)](https://huggingface.co/docs/peft/package_reference/lora)
- [Hugging Face Models save_pretrained(opens in a new tab)](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)
- [Hugging Face Text Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)

## Project Instructions

To pass this project, your code must:

1. Load a pre-trained model and evaluate its performance
2. Perform parameter-efficient fine-tuning using the pre-trained model
3. Perform inference using the fine-tuned model and compare its performance to the original model

### Getting Started

This project is fairly open-ended. As long as you follow the prescribed steps, **you may choose any appropriate PEFT technique, model, evaluation approach, and fine-tuning dataset**.

- **PEFT technique**
    - The PEFT technique covered in this course was LoRA, but new techniques are continuously being developed. See the [PEFT README(opens in a new tab)](https://github.com/huggingface/peft) for links to the papers behind each of the supported techniques.
    - If you are unsure, we recommend using **LoRA** as your PEFT technique. LoRA is the only PEFT technique that is compatible with all models at this time.
- **Model**
    - Your choice of model will depend on your choice of PEFT technique.
    - Unless you plan to use your own hardware/GPU rather than the Udacity Workspace, it's best to choose a smaller model.
    - The model must be compatible with a sequence classification task.
    - If you are unsure, we recommend using **GPT-2** as your model. This is a relatively small model that is compatible with sequence classification and LoRA.

For specific model names in the Hugging Face registry, you can use the widget at the bottom of the [PEFT documentation homepage(opens in a new tab)](https://huggingface.co/docs/peft/index) (select "sequence classification" from the drop-down).

- **Evaluation approach**
    - The evaluation approach covered in this course was the `evaluate` method with a Hugging Face `Trainer`. You may use the same approach, or any other reasonable evaluation approach for a sequence classification task
    - The key requirement for the evaluation is that you must be able to compare the original foundation model's performance and the fine-tuned model's performance.
- **Dataset**
    - Your PEFT process must use a dataset from Hugging Face's `datasets` library. As with the selection of model, you will need to ensure that the dataset is small enough that it is usable in the Udacity Workspace.
    - The key requirement for the dataset is that it matches the task. Follow this link to [view Hugging Face datasets filtered by the text classification task(opens in a new tab)](https://huggingface.co/datasets?task_categories=task_categories:text-classification)

### Loading and Evaluating a Foundation Model

#### Loading the model

Once you have selected a model, load it in your notebook.

#### Evaluating the model

Perform an initial evaluation of the model on your chosen sequence classification task. This step will require that you also load an appropriate tokenizer and dataset.

### Performing Parameter-Efficient Fine-Tuning

#### Creating a PEFT config

Create a PEFT config with appropriate hyperparameters for your chosen model.

#### Creating a PEFT model

Using the PEFT config and foundation model, create a PEFT model.

#### Training the model

Using the PEFT model and dataset, run a training loop with at least one epoch.

#### Saving the trained model

Depending on your training loop configuration, your PEFT model may have already been saved. If not, use `save_pretrained` to save your progress.

### Performing Inference with a PEFT Model

#### Loading the model

Using the appropriate PEFT model class, load your trained model.

#### Evaluating the model

Repeat the previous evaluation process, this time using the PEFT model. Compare the results to the results from the original foundation model.

## General Questions

### 1. What is the goal of this project?

The project focuses on applying **parameter-efficient fine-tuning (PEFT)** to pre-trained models using the Hugging Face `peft` library. Participants will:

- Load and evaluate a pre-trained model
- Apply PEFT to fine-tune the model efficiently
- Perform inference and compare performance before and after fine-tuning

### 2. What PEFT technique should I use?

The course covers **LoRA (Low-Rank Adaptation)**, and we recommend using it unless you have experience with other techniques. LoRA is compatible with most models at this time.

### 3. Which model should I choose?

- The model must be compatible with **sequence classification**.
- If unsure, we **recommend using GPT-2**, as it is lightweight and works well with LoRA.
- Using a **very large or newer model** might cause unexpected errors.

### 4. What dataset should I use?

Your dataset must be from the Hugging Face `datasets` library and must match the **text classification task**. You can explore available datasets [here(opens in a new tab)](https://huggingface.co/datasets?task_categories=text-classification).

## Troubleshooting Common Issues

### 5. Why does my saved LoRA model evaluate differently after reloading?

**Issue:** After saving a LoRA model using `save_pretrained` and reloading it with `AutoPeftModelForCausalLM`, the evaluation results are different from before the save.

**Solution:** Ensure that:

- You correctly save both the **PEFT adapter** and the **base model**.
- When reloading, you properly **merge the adapter** back into the base model before evaluating.

Steps to correctly save and reload:

`peft_model.save_pretrained("my_model")  # Save the model # Reloading the model from peft import AutoPeftModelForCausalLM reloaded_model = AutoPeftModelForCausalLM.from_pretrained("my_model")`

If the issue persists, check the [Knowledge Hub(opens in a new tab)](https://knowledge.udacity.com/?nanodegree=nd608&page=1&project=2049&rubric=5272) for additional guidance.

---

### 6. I encountered an error with `get_peft_model(base_model, dora_config)`.

**Issue:** Bug when running `model = get_peft_model(base_model, dora_config)`.

**Solution:**

- If you are using GPT-2 as **recommended**, this error should not occur.
- If using a newer or different model, ensure it is compatible with LoRA.
- Some newer models may **not fully support PEFT methods yet**, leading to compatibility issues.

To avoid problems, stick to **GPT-2** or another model explicitly mentioned as compatible in the Hugging Face PEFT documentation.

---

### 7. My model's performance does not improve after fine-tuning.

**Issue:** The fine-tuned model does not perform significantly better than the base model.

**Solution:**

- Ensure your dataset is appropriate for the task and large enough to learn meaningful patterns.
- Try adjusting hyperparameters such as:
    - **Learning rate**: Start with `2e-5` and adjust.
    - **Batch size**: Ensure it is not too small or too large.
    - **Number of epochs**: Try increasing to at least `3` if the dataset allows.
- Consider **checking logs and loss curves** to ensure proper convergence.

---

### 8. The model runs out of memory (OOM error) during fine-tuning.

**Issue:** Running out of memory, especially on **limited hardware** like Udacity Workspace.

**Solution:**

- Use a **smaller model** (e.g., GPT-2 instead of larger models like GPT-3 or BERT-Large).
- Reduce **batch size** (e.g., set `per_device_train_batch_size=1`).
- Enable **gradient checkpointing** to reduce memory usage:
    
    `base_model.gradient_checkpointing_enable()`
    
- If using LoRA, ensure **only a subset of layers is being fine-tuned**.

---

### 9. How do I compare the original and fine-tuned models effectively?

**Issue:** Not sure how to evaluate and compare model performance.

**Solution:**

- Use the Hugging Face `Trainer` and `evaluate` method.
- Save evaluation results of both models and **compare key metrics (e.g., accuracy, F1-score)**.
- Example code:
    
    `original_performance = trainer.evaluate() fine_tuned_performance = fine_tuned_trainer.evaluate() print("Original Model:", original_performance) print("Fine-Tuned Model:", fine_tuned_performance)`
    

---

### 10. Which files should I include when submitting my project?

**Issue:** The project instructions mention including the notebook and saved weights, but I’m unsure which files are the saved weights.

**Solution:**

- If you used `save_pretrained`, the saved weights will typically be in a directory (e.g., `my_model/`) containing files like:
    - `pytorch_model.bin` (model weights)
    - `config.json` (model configuration)
    - `adapter_config.json` (for PEFT models)
- If your model is stored as a .zip file, ensure it includes both the notebook and the folder with these weight files.
- If unsure, check the directory where you saved the model using `os.listdir("my_model/")`.

---

### 11. How can I prevent the workspace from crashing due to large files?

**Issue:** If the total file size in /workspace exceeds 1GB, the workspace may crash and fail to reload. This issue is often caused by large model weight files or intermediate checkpoint files generated automatically during training.

**Solution:**

#### 1. Move model checkpoint files to another location

Instead of saving checkpoints in `/workspace`, store them in a temporary directory like `/tmp`. For example,

`import torch torch.save(model.state_dict(), "/tmp/my_model_checkpoint.pth")`

#### 2. Limit the number of saved checkpoints

By default, PyTorch’s `torch.save()` function saves a new checkpoint every time, quickly consuming storage. You can limit the number of saved checkpoints by maintaining only the latest ones. For example,

`import os import torch checkpoint_dir = "/workspace/checkpoints" os.makedirs(checkpoint_dir, exist_ok=True) # Save checkpoint checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_latest.pth") torch.save(model.state_dict(), checkpoint_path) # Keep only the last 3 checkpoints checkpoints = sorted(os.listdir(checkpoint_dir), reverse=True) if len(checkpoints) > 3:     os.remove(os.path.join(checkpoint_dir, checkpoints[-1]))  # Delete the oldest checkpoint`

#### 3. Automatically delete unnecessary files after training

If you no longer need old checkpoints, delete them after training to free up space. For example,

`import os file_path = "/workspace/old_checkpoint.pth" if os.path.exists(file_path):     os.remove(file_path)  # Delete file to free up space`

---

# Project: Apply Lightweight Fine-Tuning to a Foundation Model

## Prepare the Foundation Model

| Criteria                      | Submission Requirements                                                                                                                                                                                                                                                                                           |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Load a pretrained HF model    | Includes the relevant imports and loads a pretrained Hugging Face model that can be used for sequence classification                                                                                                                                                                                              |
| Load and preprocess a dataset | Includes the relevant imports and loads a Hugging Face dataset that can be used for sequence classification. Then includes relevant imports and loads a Hugging Face tokenizer that can be used to prepare the dataset.<br><br>A subset of the full dataset may be used to reduce computational resources needed. |
| Evaluate the pretrained model | At least one classification metric is calculated using the dataset and pretrained model                                                                                                                                                                                                                           |

## Perform Lightweight Fine-Tuning

|Criteria|Submission Requirements|
|---|---|
|Create a PEFT model|Includes the relevant imports, initializes a Hugging Face PEFT config, and creates a PEFT model using that config|
|Train the PEFT model|The model is trained for at least one epoch using the PEFT model and dataset|
|Save the PEFT model|Fine-tuned parameters are saved to a separate directory. The saved weights directory should be in the same home directory as the notebook file.|

## Perform Inference Using the Fine-Tuned Model

|Criteria|Submission Requirements|
|---|---|
|Load the saved PEFT model|Includes the relevant imports then loads the saved PEFT model|
|Evaluate the fine-tuned model|Repeats the earlier evaluation process (same metric(s) and dataset) to compare the fine-tuned version to the original version of the model|

### Suggestions to Make Your Project Stand Out

1. Try using the `bitsandbytes` package (installed in the workspace) to combine quantization and LoRA. This is also known as QLoRA
2. Try training the model using different PEFT configurations and compare the results
