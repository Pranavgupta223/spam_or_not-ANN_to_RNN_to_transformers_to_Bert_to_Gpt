from dataset.data_preprocessing import get_dataset

from sklearn.model_selection import train_test_split
def train_test_data():
    df = get_dataset()
    X = df['clean_text']
    y = df['label']
    X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    return X_train , X_test , y_train , y_test
