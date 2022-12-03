# MomentoRetroalimentacionModulo3

Note: I'd suggest using the code from the Google Drive link I added on the assignment's submission to avoid having to add the datasets manually.

The most important step to take to be able to run the code is to add the datasets to the directory, they have to be manually added because they surpass Github's file size limit. They have to be added to the project root directory like such:
```
MomentoRetroalimentacionModulo3
│   README.md
│   Python files
|   europarl-v7.es-en.es
|   europarl-v7.es-en.en
│
└───NERData
│   │   dev
│   │   train
|   |   test
|   |   train_trunc
```

The datasets can be found on the Google Drive link I added on the assignment's submission.

Afterwards install the dependencies with:

```bash
pip install -r requirements.txt
```

And then run with:
```bash
python run.py
```
No API Keys needed.
![alt text](training.png "Training Graph")
![alt text](weights.png "Weights Graph")
