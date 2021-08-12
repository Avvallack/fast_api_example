import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd


def preprocess_data(data_path: str) -> tuple:
    df = pd.read_csv(data_path)
    X = df.drop(columns='target')
    y = df['target'].values
    scaler = StandardScaler()
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, stratify=y)
    scaled_X = scaler.fit_transform(train_X)
    with open('./data/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    return {'train_set': scaled_X, 'train_target': train_y, 'scaler': scaler, 'test_data': (test_X, test_y)}


def train_model(X: pd.DataFrame, y: pd.DataFrame):
    model = LinearRegression()
    model.fit(X, y)
    with open('./data/model.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    prepared_data = preprocess_data('./data/wine.csv')
    train_model(prepared_data['train_set'], prepared_data['train_target'])
