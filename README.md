# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Libraries Required:

Please ensure the following libraries are installed in your environment:
1. pandas
2. nltk
3. flask
4. sklearn
5. sqllchemy
6. plotly

### File Descriptions:
- app
    - template
        - master.html  # main page of web app
        - go.html  # classification result page of web app
    - run.py  # Flask file that runs app

- data
    - disaster_categories.csv  # data to process 
    - disaster_messages.csv  # data to process
    - process_data.py
    - YourDatabaseName.db   # IGNORE 
    - DisasterResponse.db   # This db contains a table categorised messages with the clean data

- models
    - train_classifier.py
    - classifier.pkl  # saved model 

- README.md

