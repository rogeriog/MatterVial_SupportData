# Save this file as "extract_here.py"

import requests
import tarfile
import io
import os

# --- Configuration ---
# The direct download URL for the .tar.gz file.
FILE_URL = "https://figshare.com/ndownloader/files/57963316"
# ---------------------

def download_and_extract(url: str):
    """Downloads and extracts a .tar.gz file to the current directory."""
    print(f"▶️  Starting download from: {url}")

    # Step 1: Download the file content into memory
    try:
        response = requests.get(url, stream=True, timeout=60)
        # Check for download errors (like 404 Not Found)
        response.raise_for_status()
        file_in_memory = io.BytesIO(response.content)
        print("✅ Download complete.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Failed to download the file. Details: {e}")
        return

    # Step 2: Extract the .tar.gz file to the current folder
    print("📦 Extracting files here...")
    try:
        # Open the gzipped tar archive directly from the in-memory object
        with tarfile.open(fileobj=file_in_memory, mode='r:gz') as tar:
            # The "." tells it to extract to the current directory
            tar.extractall(path=".")
        print(f"🎉 Success! Files extracted to '{os.path.abspath('.')}'")
    except tarfile.TarError as e:
        print(f"❌ Error: Failed to extract the file. It may be corrupt. Details: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during extraction: {e}")

if __name__ == "__main__":
    download_and_extract(FILE_URL)