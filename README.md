---
# license: apache-2.0
# tags:
# - generated_from_trainer
# metrics:
# - f1
# - accuracy
# model-index:
# - name: flan-t5-base-imdb-text-classification
#   results: 
#   - task:
#       name: Sequence-to-sequence Language Modeling
#       type: text2text-generation
#     dataset:
#       name: imdb
#       type: imdb
#       config: imdb
#       split: test
#       args: imdb
#     metrics:
#     - name: Accuracy
#       type: accuracy
#       value: 93.0000
# datasets:
# - imdb
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# flan-t5-base-imdb-text-classification

This model is a fine-tuned version of [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) on the [IMDB](https://huggingface.co/datasets/imdb) dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0767
- F1: 95.084
- Gen Len: 2.4976

```cmd
              precision    recall  f1-score   support

           0       0.97      0.88      0.92     12500
           1       0.89      0.97      0.93     12500

    accuracy                           0.93     25000
   macro avg       0.93      0.93      0.93     25000
weighted avg       0.93      0.93      0.93     25000
```

## Model description
In this implementation, using the **Flan T5 large language model**, we performed the Text Classification task on the IMDB dataset and obtained a very good **accuracy of 93%**.


## Training and evaluation data
This model was trained on the imdb train dataset with 25,000 data and then tested and evaluated on the imdb test dataset with 25,000 data.

## Usage

1. Install dependencies
```python
!pip install transformers==4.28.1 datasets
```

2. Load IMDB Corpus
```python
from datasets import load_dataset

dataset_id = "imdb"

# Load dataset from the hub
dataset = load_dataset(dataset_id)
```

3. Load fine tune flan t5 model
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("mohammadtaghizadeh/flan-t5-base-imdb-text-classification")
model = AutoModelForSeq2SeqLM.from_pretrained("mohammadtaghizadeh/flan-t5-base-imdb-text-classification")
model.to('cuda')
```

4. Test the model
```python
from tqdm.auto import tqdm

samples_number = len(dataset['test'])
progress_bar = tqdm(range(samples_number))
predictions_list = []
labels_list = []
for i in range(samples_number):
  text = dataset['test']['text'][i]
  inputs = tokenizer.encode_plus(text, padding='max_length', max_length=512, return_tensors='pt').to('cuda')
  outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150, num_beams=4, early_stopping=True)
  prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
  predictions_list.append(prediction)
  labels_list.append(dataset['test']['label'][i])

  progress_bar.update(1)
```

5. Classification report
```python
from sklearn.metrics import classification_report

str_labels_list = []
for i in range(len(labels_list)): str_labels_list.append(str(labels_list[i]))

report = classification_report(str_labels_list, predictions_list)
print(report)
```

Output
```cmd
              precision    recall  f1-score   support

           0       0.97      0.88      0.92     12500
           1       0.89      0.97      0.93     12500

    accuracy                           0.93     25000
   macro avg       0.93      0.93      0.93     25000
weighted avg       0.93      0.93      0.93     25000
```


## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 2

### Training results

| Training Loss | Epoch | Step |
|:-------------:|:-----:|:----:|
| 0.100500      | 1.0   | 3125 |
| 0.043600      | 2.0   | 6250 | 

### Framework versions

- Transformers 4.28.1
- Pytorch 2.0.0+cu118
- Datasets 2.12.0
- Tokenizers 0.13.3