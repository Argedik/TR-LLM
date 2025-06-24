import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

# Türkçe veri seti örneği (Wikipedia veya başka bir açık veri seti kullanılabilir)
dataset = load_dataset("bilgeyik/turkish-llm-dataset", split="train")

# Model ve tokenizer seçimi (örnek: tiny model)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="../models/turkish-llm",
    evaluation_strategy="steps",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_steps=1000,
    logging_steps=100,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("../models/turkish-llm")
    tokenizer.save_pretrained("../models/turkish-llm")
