import requests
import os

def download_files_from_github_folder(repo_url, folder_path):
    # Create a GET request to fetch the contents of the folder
    response = requests.get(f"{repo_url}")

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response JSON
        files = response.json()

        # Create a directory to save the downloaded files
        os.makedirs(folder_path, exist_ok=True)

        # Download each file
        for file in files:
            # Get the download URL for the file
            download_url = file["download_url"]

            # Send a GET request to download the file
            file_response = requests.get(download_url)

            # Extract the filename from the URL
            filename = download_url.split("/")[-1]

            # Save the file to the directory
            with open(os.path.join(folder_path, filename), "wb") as f:
                f.write(file_response.content)

        print("Files downloaded successfully.")
    else:
        print("Error: Failed to fetch folder contents.")

# Example usage
repository_url = "https://api.github.com/repos/furnya/bn_classifier_data/contents/data"
folder_path = "./Images"

download_files_from_github_folder(repository_url, folder_path)
