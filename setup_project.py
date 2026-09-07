import os

directories = [
    "config", "core", "input", "validation", "quality", 
    "preprocessing", "morphology", "diagnosis", "aggregation", 
    "uncertainty", "explainability", "reporting", "traceability", 
    "gui", "tests"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    init_file = os.path.join(directory, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write(f'"""{directory.capitalize()} module initialization."""\n')

print("NECK-CAD foundational directory structure created successfully.")