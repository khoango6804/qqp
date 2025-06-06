import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline,DataCollatorWithPadding
from pathlib import Path
from datasets import load_dataset,DatasetDict
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer


model_name =Path(r"K:\checkpoint-116000")
num_labels = 2
id2label = {0: "not duplicates", 1: "duplicates"}
label2id = {"not duplicates": 0, "duplicates": 1}

model = AutoModelForSequenceClassification.from_pretrained(
model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    trust_remote_code=True,
    local_files_only=True )
tokenizer = AutoTokenizer.from_pretrained(model_name
                                          ,local_files_only=True, trust_remote_code=True)

def load_parquet_dataset(train_path, val_path, test_path):
    dataset = load_dataset('parquet', data_files={
        'train': train_path,
        'validation': val_path,
        'test': test_path
    })
    return dataset['train'], dataset['validation'], dataset['test']

train_path = r"K:\checkpoint-116000\qqp\train-00000-of-00001.parquet"
val_path = r"K:\checkpoint-116000\qqp\validation-00000-of-00001.parquet"
test_path = r"K:\checkpoint-116000\qqp\test-00000-of-00001.parquet"

# Load datasets
train_dataset, val_dataset, test_dataset = load_parquet_dataset(train_path, val_path, test_path)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

def format_prompt(question1, question2):
    return f"Question 1: {question1}\nQuestion 2: {question2}"

def preprocess_function(examples):
    texts = [format_prompt(q1, q2) for q1, q2 in zip(examples["question1"], examples["question2"])]
    tokenized = tokenizer(texts, truncation=True, padding=True, max_length=256)
    tokenized["labels"] = examples["label"]
    return tokenized

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)


training_args = TrainingArguments(
    output_dir="QQP_classifier",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    lr_scheduler_type="cosine",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    bf16=True,  
    load_best_model_at_end=True,
    push_to_hub=False,  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()