import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
import pickle
import re
import nltk

nltk.download(['punkt', 'wordnet'])

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    """This function loads the processed data from the supplied database filepath
    
    Input:
        1- database_filepath: location of the sql lite file where ML data is present
     
    Output:
        1- X: this is the message column of df
        2- Y: this is the target variable columns
        3- Y.columns: this is a list of all target variables
    """
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('categorised_messages', engine) 

    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    return X, Y, Y.columns


def tokenize(text):
    """This function breaks texts into tokens. Also replaces URLs and lemmatizes the tokens
    for consistency.
    
    Input:
        1- Text: Sentence of paragraph
    Output:
        1- List of tokens
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """This is the main training function. It uses a pipeline object 
    to group together CountVectorizer, TfidfTransformer, and 
    MultiOutputClassifier which uses RandomForestClassifier.
    
    It also makes use of the GridSearch of vect__max_features but 
    this can be changed to improve the model.
    """
    # create model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # specify parameters for grid search
    parameters = {
        'vect__max_features': [None, 100, 1000]
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """This function is used for model evaluation.
    """
    y_pred = model.predict(X_test)
    
    for i, column in enumerate(category_names):
        print("Report for target column: " + column)
        print(classification_report(Y_test.values[:, i], y_pred[:,i], target_names=["0", "1"]))


def save_model(model, model_filepath):
    """Function to save the trained model as a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """Main training function that loads, trains and saves the trained classifier.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()