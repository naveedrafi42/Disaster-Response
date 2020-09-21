import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """This function loads messages and category data, left joins the two 
    based on 'id' and returns a pandas dataframe.
    
    The function will raised an error if input is invalid or data doesn't exist.
    
    Input:
        messages_filepath: location of messages data file from the project root
        categories_filepath: location of categories data file from the project root
        
    Output:
        df: Dataframe containing the joined dataset.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on="id", how="left")
    
    return df


def clean_data(df):
    """This function cleans and prepares the raw data set into 
    one that is easier to work with. Also drops duplicates.
    
    Input:
        df: Dataframe containing raw, joined data
        
    Output:
        df: Dataframe containing one column per category using
        '0', '1', '2' as flags to indicate whether or not the 
        message belongs to category mentioned as column name
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    slicer = lambda x: x[:-2]
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [slicer(i) for i in list(row)]
    
    categories.columns = category_colnames
    
    slice_last_character = lambda x: int(x[-1])
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [slice_last_character(i) for i in list(categories[column])]
    
    # recreate df with new category data
    df = df.drop(['categories'], axis=1)   
    df = pd.merge(df, categories, left_index=True, right_index=True)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """This function saves the prepared data into an sqllite database file
    
    Input:
        df: Dataframe with clean data
        database_filename: path of where to store the data for model ingestion
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('categorised_messages', engine, index=False)


def main():
    """Main ETL function that loads, processes and saves the data.
    """
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()