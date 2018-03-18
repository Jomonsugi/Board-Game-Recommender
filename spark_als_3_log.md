## spark_als_3.py log
- 2017-03-11

1. The script is successfully running through saving an optimized model and predicting for one user. The data frame gained by the nmf model at ```nmf_labeled_df()``` is failing because it can't find the file. I will also need that file to be protocol 3 as I am running this in Python 3.

2. I need to figure out how to overwrite a saved spark model as I am hitting an error when saving as the file already exist, however have just changed the file name every run to get past that error for now

3. The model starts by reading a csv file. I could write a script for the mongodb database to just be saved as a csv file, otherwise will need to figure out the connector. It's such a pain, maybe not really worth it...

4. want to grid search alpha but getting this error:
```TypeError: Invalid param value given for param "alpha". Could not convert [0.5] to float
```

5. Discovered the UI! While job is running check out: http://localhost:4040/jobs/

6. Got the checkpointInterval working. I set to 10, maybe 20 would work, but allows for more iterations. Did 200 with no issue.

7. Only using df (not rdd) in the function ```to_user_unrated_df```, simplified the function as well 
