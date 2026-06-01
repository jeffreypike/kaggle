#!/usr/bin/env python3
import os
import sys
import json
import glob
import shutil
import subprocess
import argparse
import nbformat
from dotenv import load_dotenv

def get_project_py_files(project_dir):
    """
    Finds all helper Python files in <project_dir>/src/ recursively,
    excluding __init__.py files.
    """
    src_dir = os.path.join(project_dir, "src")
    if not os.path.isdir(src_dir):
        return []
    
    py_files = []
    for root, _, files in os.walk(src_dir):
        for file in sorted(files):
            if file.endswith(".py") and file != "__init__.py":
                py_files.append(os.path.join(root, file))
    return py_files

def bundle_notebook(notebook_path, py_files, project_dir):
    """
    Reads a notebook, prepends helper python files as code cells, and returns
    the modified nbformat object.
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    # Prepare helper cells to prepend
    new_cells = []
    for py_file in py_files:
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        rel_path = os.path.relpath(py_file, start=project_dir)
        cell_source = f"# AUTO-GENERATED: Prepended helper code from {rel_path}. Do not edit.\n\n{content}"
        
        cell = nbformat.v4.new_code_cell(source=cell_source)
        cell.metadata["prepended_from"] = rel_path
        new_cells.append(cell)
        
    nb.cells = new_cells + nb.cells
    return nb

def clean_notebook(notebook_path):
    """
    Reads a notebook and removes any code cells prepended by the sync script.
    """
    if not os.path.exists(notebook_path):
        return
        
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    clean_cells = []
    removed_count = 0
    
    for cell in nb.cells:
        is_prepended = False
        if cell.cell_type == "code":
            if cell.metadata.get("prepended_from") is not None:
                is_prepended = True
            elif cell.source.startswith("# AUTO-GENERATED: Prepended helper code"):
                is_prepended = True
                
        if is_prepended:
            removed_count += 1
        else:
            clean_cells.append(cell)
            
    if removed_count > 0:
        nb.cells = clean_cells
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"-> Removed {removed_count} prepended helper cells from local notebook: {notebook_path}")

def find_metadata_files(project_dir):
    """
    Finds all kernel-metadata.json files in <project_dir>/notebooks/ recursively.
    """
    notebooks_dir = os.path.join(project_dir, "notebooks")
    if not os.path.isdir(notebooks_dir):
        return []
    return glob.glob(os.path.join(notebooks_dir, "**/kernel-metadata.json"), recursive=True)

def handle_push(project_dir, dry_run=False):
    print(f"=== Starting push sync for project: {project_dir} ===")
    
    # 1. Find helpers
    py_files = get_project_py_files(project_dir)
    print(f"Found {len(py_files)} helper python files to prepend:")
    for py_file in py_files:
        print(f"  - {py_file}")
        
    # 2. Find notebook metadata files
    metadata_files = find_metadata_files(project_dir)
    if not metadata_files:
        print(f"No kernel-metadata.json files found in {project_dir}/notebooks/")
        return
        
    for metadata_path in metadata_files:
        print(f"\nProcessing notebook directory: {os.path.dirname(metadata_path)}")
        
        # Read metadata to find the notebook file
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        code_file = meta.get("code_file")
        if not code_file:
            print(f"Warning: 'code_file' not specified in {metadata_path}. Skipping.")
            continue
            
        notebook_dir = os.path.dirname(metadata_path)
        notebook_path = os.path.join(notebook_dir, code_file)
        
        if not os.path.exists(notebook_path):
            print(f"Error: Notebook file {notebook_path} does not exist. Skipping.")
            continue
            
        # Bundle helper scripts
        print(f"Bundling helpers into {notebook_path}...")
        nb_bundled = bundle_notebook(notebook_path, py_files, project_dir)
        
        # Create output build path
        rel_notebook_dir = os.path.relpath(notebook_dir, start=".")
        build_dir = os.path.join(".build", rel_notebook_dir)
        os.makedirs(build_dir, exist_ok=True)
        
        build_notebook_path = os.path.join(build_dir, code_file)
        build_metadata_path = os.path.join(build_dir, "kernel-metadata.json")
        
        # Write bundled notebook & copy metadata to build folder
        with open(build_notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb_bundled, f)
            
        shutil.copy(metadata_path, build_metadata_path)
        print(f"Bundled notebook written to {build_notebook_path}")
        
        if dry_run:
            print(f"[DRY-RUN] Would run: kaggle kernels push -p {build_dir}")
        else:
            print(f"Pushing to Kaggle from build dir: {build_dir}")
            try:
                result = subprocess.run(
                    ["kaggle", "kernels", "push", "-p", build_dir],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error pushing to Kaggle:\n{e.stderr}", file=sys.stderr)
                sys.exit(1)

def handle_pull(project_dir):
    print(f"=== Starting pull sync for project: {project_dir} ===")
    
    metadata_files = find_metadata_files(project_dir)
    if not metadata_files:
        print(f"No kernel-metadata.json files found in {project_dir}/notebooks/")
        return
        
    for metadata_path in metadata_files:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        kernel_slug = meta.get("id")
        code_file = meta.get("code_file")
        notebook_dir = os.path.dirname(metadata_path)
        notebook_path = os.path.join(notebook_dir, code_file)
        
        if not kernel_slug:
            print(f"Error: 'id' (slug) not specified in {metadata_path}. Skipping.")
            continue
            
        print(f"\nPulling kernel '{kernel_slug}' from Kaggle into {notebook_dir}...")
        try:
            # kaggle kernels pull downloads both the notebook and creates a metadata file
            result = subprocess.run(
                ["kaggle", "kernels", "pull", "-k", kernel_slug, "-p", notebook_dir, "-m"],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            
            # Clean up the prepended code from the pulled notebook to keep local source clean
            clean_notebook(notebook_path)
        except subprocess.CalledProcessError as e:
            print(f"Error pulling from Kaggle:\n{e.stderr}", file=sys.stderr)
            sys.exit(1)

def main():
    load_dotenv()
    
    # Kaggle CLI v2 compatibility
    if "KAGGLE_API_TOKEN" not in os.environ and "KAGGLE_KEY" in os.environ:
        if os.environ["KAGGLE_KEY"].startswith("KGAT_"):
            os.environ["KAGGLE_API_TOKEN"] = os.environ["KAGGLE_KEY"]
            
    parser = argparse.ArgumentParser(description="Kaggle Monorepo Notebook Sync & Bundler")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Push command
    push_parser = subparsers.add_parser("push", help="Bundle helper scripts and push notebooks to Kaggle")
    push_parser.add_argument("project_dir", help="The project directory to sync (e.g. stellar_class)")
    push_parser.add_argument("--dry-run", action="store_true", help="Perform bundling but do not push to Kaggle")
    
    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull latest notebooks from Kaggle and strip bundled helpers")
    pull_parser.add_argument("project_dir", help="The project directory to sync (e.g. stellar_class)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.project_dir):
        print(f"Error: {args.project_dir} is not a valid directory.", file=sys.stderr)
        sys.exit(1)
        
    if args.command == "push":
        handle_push(args.project_dir, dry_run=args.dry_run)
    elif args.command == "pull":
        handle_pull(args.project_dir)

if __name__ == "__main__":
    main()
