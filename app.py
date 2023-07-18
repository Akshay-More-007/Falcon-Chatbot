from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/starcoder"
device = "cpu" # for GPU usage or "cpu" for CPU usage
api_key = "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"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_auth_token=api_key)
model = AutoModelForCausalLM.from_pretrained(checkpoint, use_auth_token=api_key).to(device)

inputs = tokenizer.encode("def print_hello_world ():", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
