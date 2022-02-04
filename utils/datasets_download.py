import zipfile
import requests


def movielens_downloader(data_name='ml-100k') -> None:
    """
    downloads the movielens dataset and saves it to DATA_PATH/movielens folder.
    :param data_name: the dataset name.
                      Here: http://files.grouplens.org/datasets/movielens/ you can find different datasets of movielens.
                      For example: ml-100k, ml-10m, etc.
    """
    movielens_path = 'data/movielens/'

    url = f'http://files.grouplens.org/datasets/movielens/{data_name}.zip'
    zip_path = movielens_path + f'{data_name}.zip'

    response = requests.get(url, allow_redirects=True)

    if response.status_code != 200:
        raise ConnectionError('error downloading movielens dataset')
    else:
        print(f'successfully downloaded {data_name}')

    open(zip_path, 'wb').write(response.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(movielens_path)

if __name__ == '__main__':
    movielens_downloader()

