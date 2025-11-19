import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

model_id = "microsoft/deberta-v3-small"
onnx_path = Path("deberta-v3-small.onnx")
quantized_path = Path("deberta-v3-small-int8.onnx")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

# Dummy input for tracing
inputs = tokenizer("Hello world!", return_tensors="pt")

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    args=(
        inputs["input_ids"],
        inputs["attention_mask"],
    ),
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

print(f"Saved FP32 ONNX model → {onnx_path}")

print("Starting INT8 dynamic quantization...")
quantize_dynamic(
    model_input=str(onnx_path),
    model_output=str(quantized_path),
    weight_type=QuantType.QInt8,  # this is INT8
)

print(f"Saved INT8 quantized model → {quantized_path}")
