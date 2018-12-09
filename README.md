# Disaster Response Pipeline Project

### Instructions:

#### process data and save it to sqlite DB
```bash
cd data 
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```

#### Train and save the model
```bash
cd models
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```

#### flask
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
