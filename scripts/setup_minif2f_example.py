import json
from datasets import load_dataset

# Load the miniF2F dataset
dataset = load_dataset("cat-searcher/minif2f-lean4")

# Export the validation split as a standard JSON array
split_name = 'validation' if 'validation' in dataset else 'valid'
with open("data/val.json", "w") as f:
    json.dump(list(dataset[split_name]), f, indent=4)

# Export the test split as a standard JSON array
with open("data/test.json", "w") as f:
    json.dump(list(dataset["test"]), f, indent=4)

print("Successfully saved miniF2F as standard JSON arrays!")
