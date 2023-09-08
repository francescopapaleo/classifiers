# pip install nbformat nbconvert

import os
import nbformat
from tensorboard import notebook

# Get the current path
repository_path = os.path.dirname(os.path.realpath(__file__))

# List of file extensions for Jupyter notebooks
notebook_extensions = ['.ipynb']

# Set to store dependencies
dependencies = set()

def extract_dependencies(notebook_path):
    """Extract dependencies from a Jupyter notebook"""
    with open(notebook_path, 'r') as nb_file:
        notebook = nbformat.read(nb_file, as_version=4)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                code = cell['source']
                dependencies.update(
                    line.split()[1] for line in code.split('\n') 
                    if line.startswith('import') or line.startswith('from')
                )

# Recursively scan the repository for Jupyter notebooks

for root, dirs, files in os.walk(repository_path):
    for file in files:
        if file.endswith(tuple(notebook_extensions)):
            notebook_path = os.path.join(root, file)
            extract_dependencies(notebook_path)
    

# Write the dependencies to a requirements.txt file
with open('requirements.txt', 'w') as req_file:
    req_file.write('\n'.join(sorted(dependencies)))

print('Requirements saved to requirements.txt')
