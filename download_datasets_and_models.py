import requests

def download_figshare_file(file_url, save_path):
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(save_path, 'w') as file:
            file.write(response.content)
        print(f'Downloaded: {save_path}')
    else:
        print(f'Failed to download: {response.status_code}')

# Example usage
figshare_file_url = 'https://figshare.com/ndownloader/articles/27132093?private_link=ad92db8097ddc8d901f5'
save_path = 'datasets_and_models'  # Specify the name and extension for the saved file

download_figshare_file(figshare_file_url, save_path)
