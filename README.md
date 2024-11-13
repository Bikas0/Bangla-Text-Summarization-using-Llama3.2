<h1>Bangla Text Summarization using Llama 3.2</h1>

This repository contains code and resources for performing text summarization on Bangla (Bengali) text data using the Llama 3.2 1B model, fine-tuned with the help of the Unsloth library. The goal of this project is to generate concise, coherent summaries for Bangla texts using a state-of-the-art language model fine-tuned for optimal performance on the Bangla language.

Table of Contents
Background
Requirements
Installation
Data Preparation
Fine-tuning the Model
Evaluation
Usage
Contributing
License
Background
With the growing volume of digital content in Bangla, an efficient text summarization tool is essential for processing and understanding large amounts of text. This project utilizes Llama 3.2, a high-performing language model, and Unsloth, a library designed for efficient fine-tuning. Together, they allow for fine-tuning the model on Bangla datasets to produce relevant and coherent summaries of Bangla texts.

Requirements

```bash
Python 3.8 or higher
CUDA (if using GPU)
transformers
torch
unsloth (for efficient fine-tuning)
scikit-learn (for evaluation metrics)
pandas
numpy
```

Installation
Clone the repository:

```bash
Copy code
git clone https://github.com/yourusername/Bangla-Text-Summarization-using-Llama3.2.git
cd Bangla-Text-Summarization-using-Llama3.2
```

Create a virtual environment (optional but recommended):

```bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the required packages:

```bash
Copy code
pip install -r requirements.txt
```

Data Preparation
To fine-tune Llama 3.2 for Bangla summarization, you need a dataset of Bangla texts with reference summaries. Ensure the dataset has two main columns: text (the full Bangla text) and summary (the target summary).

Sample dataset format (CSV):

csv
Copy code

```bash
text,summary
"সিরাম ইনস্টিটিউট বাংলাদেশে প্রতিমাসে লাখ ডোজ টিকা...", "বাংলাদেশে আগামীকাল সোমবার করোনাভাইরাসের টিকার ডোজ দেয়া দিচ্ছে কর্তৃপক্ষ"
...
```

Fine-tuning the Model
We utilize Unsloth for fine-tuning Llama 3.2. The following steps outline the process:

Prepare the dataset in a suitable format (e.g., CSV or JSON).

Run the fine-tuning script:

python
Copy code
python fine_tune.py --data_path path/to/dataset.csv --epochs <num_epochs> --batch_size <batch_size> --learning_rate <learning_rate>
Modify num_epochs, batch_size, and learning_rate based on your hardware and requirements.

The fine_tune.py script:

Loads the Llama 3.2 model and tokenizer.
Configures Unsloth for fine-tuning.
Saves the fine-tuned model in the models/ directory.
Evaluation
Once the model is fine-tuned, evaluate it using standard text summarization metrics:

ROUGE Score: Measures the overlap between generated and reference summaries.
BLEU Score: Evaluates the precision of n-grams in the generated summary.
To evaluate, run:

python
Copy code
python evaluate.py --model_path models/fine_tuned_model --data_path path/to/validation_set.csv
Usage
To generate a summary with the fine-tuned model:

Load the fine-tuned model.
Input the Bangla text to summarize:
python
Copy code

```bash
 # Now `row` is a Pandas Series representing a row of data
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Please provide a summary of the following article",  # Instruction
                row["article"],  # The article content to summarize
                ""  # Leave output blank for generation
            )
        ],
        return_tensors="pt"
    ).to("cuda")

    # Generate text and keep the output in a variable
    generated_ids = model.generate(**inputs, max_new_tokens=2024)

    # Decode the generated text and store it in a variable
    generated_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
```

Contributing
Contributions are welcome! If you have improvements, suggestions, or new features, please fork this repository and create a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
