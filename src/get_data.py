import sklearn.datasets as ds


def get_wine_dataset(save_path: str) -> None:
    dataset = ds.load_wine(as_frame=True)
    dataset['frame'].to_csv(save_path + '/wine.csv', index=False)


if __name__ == '__main__':
    get_wine_dataset('./data')
