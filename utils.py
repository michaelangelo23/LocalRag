import os
import shutil

def move_file_to_directory(source_path: str, destination_directory: str) -> bool:

    if not os.path.exists(source_path):
        print(f"Error: Source file '{source_path}' not found. Cannot move.")
        return False

    try:
        # Ensure the destination directory exists
        os.makedirs(destination_directory, exist_ok=True)

        file_name = os.path.basename(source_path)
        destination_path = os.path.join(destination_directory, file_name)

        shutil.move(source_path, destination_path)
        print(f"Successfully moved '{file_name}' to '{destination_directory}'.")
        return True
    except Exception as e:
        print(f"Error moving file '{source_path}' to '{destination_directory}': {e}")
        return False