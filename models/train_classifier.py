import sys
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from typing import Tuple, List
import pickle
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# stopwords
stopwords_list = list(set(stopwords.words('english')))


def load_data(database_filepath) -> Tuple:
    """
    read the data from sqlite and return a dataframe representation
    :param database_filepath:  sqlite database file path
    :return:
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df: pd.DataFrame = pd.read_sql(f"SELECT * FROM disaster_response", engine)

    x = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1).astype(float)

    categories = y.columns.values

    return x, y, categories


def tokenize(text: str, remove_stop: bool=True) -> List[str]:
    """
    tokenize and lemmatize the text
    :param text: text to tokenize
    :param remove_stop: remove stop words, default to True
    :return: tokenized text
    """

    tokens = nltk.word_tokenize(text)

    lemmatizer = nltk.WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(x).lower().strip() for x in tokens]

    if remove_stop:
        tokens = [token for token in tokens if token not in stopwords_list]

    return tokens


def build_model() -> GridSearchCV:

    """
    Build a pipeline for the model and returns a grid search for Random Forest Classifier
    :return: grid search model
    """
    # Pipeline for transforming data, fitting to model and predicting
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())])

    # Parameters for GridSearch (simplified due to time challenges :-( )
    parameters = {
        'clf__min_samples_split': [5, 10, 15],
        'clf__n_estimators':      [50, 100, 150]}

    # GridSearch with the above parameters
    cv = GridSearchCV(
        pipeline, param_grid=parameters,
        scoring='accuracy', verbose=2,
        n_jobs=4, cv=3
    )

    return cv


def evaluate_model(model: GridSearchCV, x_test: pd.DataFrame, y_test: pd.DataFrame, category_names: List) -> None:
    """
    evaluate the model accuracy and print classification report
    :param model: GridSearchCV model to evaluate
    :param x_test: x test data
    :param y_test: y test data
    :param category_names: list of categories
    :return: None
    """

    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred, target_names=category_names))

    for idx, cat in enumerate(y_test.columns.values):
        print("{} --> {}".format(cat, accuracy_score(y_test.values[:, idx], y_pred[:, idx])))

    print('Accuracy Score: {}'.format(np.mean(y_test.values == y_pred)))


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    """
    save the model to the designated model_filepath
    :param model: GridSearchCV model to be saved
    :param model_filepath: the file path for the saved model
    :return: None
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        x, y, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(x_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, x_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
