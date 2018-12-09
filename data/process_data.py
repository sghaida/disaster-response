import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    reads messages and categories files to a dataframe
    :param messages_filepath: the path of the messages file
    :param categories_filepath: the path of categories file
    :return: dataframe of categories and messages after cleanup and concatenation
    """
    messages = pd.read_csv(messages_filepath)
    categories = fix_categories(pd.read_csv(categories_filepath))

    return pd.concat([messages, categories], join="inner", axis=1)


def fix_categories(categories: pd.DataFrame) -> pd.DataFrame:
    """
    change categorical data from string representation to numerical representation 0 / 1
    :param categories: categories DataFrame
    :return: encoded categories DataFrame
    """
    def convert_to_numerical(cats: pd.DataFrame) -> pd.DataFrame:

        for col in cats:
            cats[col] = cats[col].map(
                lambda x: 1 if int(x.split("-")[1]) > 0 else 0)
        return cats

    categories = categories['categories'].str.split(';', expand=True)
    row = categories.iloc[[1]].values[0]
    categories.columns = [x.split("-")[0] for x in row]
    categories = convert_to_numerical(categories)
    return categories


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    cleanup data by dropping duplicates
    :param df:
    :return: dataframe
    """
    return df.drop_duplicates()


def save_data(df: pd.DataFrame, database_filename: str):
    """
    save the dataframe to database
    :param df: dataframe to be saved
    :param database_filename: sqlite database file path
    :return:
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_response', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))

        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
