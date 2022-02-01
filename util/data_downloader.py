import zipfile
import requests

def movielens_data():
    # # download the MovieLens dataset
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    zip_name = '../data/movielens/ml-100k.zip'
    data_dir = 'ml-100k/'

    r = requests.get(url, allow_redirects=True)
    open(zip_name, 'wb').write(r.content)

    with zipfile.ZipFile(zip_name,"r") as zip_ref:
        zip_ref.extractall()



movielens_data()