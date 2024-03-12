import nbformat
from nbformat import NotebookNode


def write_cells_to_py(notebook_path: str, output_file: str, marker: str = "# WRITE_TO_PY"):
    """
    Writes the content of cells marked with a specific keyword to a .py file.

    :param notebook_path: Path to the Jupyter notebook file (.ipynb).
    :param output_file: Path to the output .py file.
    :param marker: The marker or keyword to look for in cell content.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        print(f'Opened {notebook_path}')
        nb = nbformat.read(f, as_version=4)

    with open(output_file, 'w', encoding='utf-8') as f:
        count = 0
        for cell in nb.cells:
            if cell.cell_type == 'code':
                # Check if the marker is in the first line of the cell
                lines = cell.source.split('\n')
                if lines and lines[0].strip().startswith(marker):
                    # Write the cell content without the marker
                    f.write('\n'.join(lines[1:]) + '\n\n')
                    count += 1
        print(f"Wrote {count} cells to {output_file}")


if __name__ == "__main__":
    write_cells_to_py('implementing-transformers/02_Original_Transformer.ipynb',
                      'implementing-transformers/02_Transformer.py', marker="# WRITE_TO_PY")
    write_cells_to_py('fine-tuning/trainer.ipynb',
                      'fine-tuning/trainer.py', marker="#trainer.py")
