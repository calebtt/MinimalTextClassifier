import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

model_id = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)  # Binary classification

# Load examples from text files for easy editing
with open('positive_examples.txt', 'r') as f:
    positive_examples = [line.strip() for line in f if line.strip()]

with open('negative_examples.txt', 'r') as f:
    negative_examples = [line.strip() for line in f if line.strip()]

data = {
    "text": positive_examples + negative_examples,
    "labels": [1] * len(positive_examples) + [0] * len(negative_examples)
}
dataset = Dataset.from_dict(data).train_test_split(test_size=0.2)  # 80/20 split for training/eval

def preprocess(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(preprocess, batched=True)

# Fine-tuning setup
training_args = TrainingArguments(
    output_dir="fine_tuned_deberta_wake",
    num_train_epochs=5,  # Increased for better learning
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    learning_rate=2e-5,
    weight_decay=0.01,
)

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

# Export to ONNX
onnx_path = Path("deberta_v3_small_fine_tuned.onnx")
quantized_path = Path("deberta_v3_small_fine_tuned_int8.onnx")

dummy_input = tokenizer("Dummy input for export", return_tensors="pt")
torch.onnx.export(
    model,
    args=(dummy_input["input_ids"], dummy_input["attention_mask"]),
    f=str(onnx_path),
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch"}
    },
    opset_version=17,
)

# Quantize to INT8
quantize_dynamic(str(onnx_path), str(quantized_path), weight_type=QuantType.QInt8)

print(f"Fine-tuned INT8 model saved to {quantized_path}")