# step 1 
# import libraries
import boto3, re, sys, math, json, os, ml.sagemaker as sagemaker
from ml.sagemaker import get_execution_role
import numpy as np                                
import pandas as pd                               
import matplotlib.pyplot as plt                   
from IPython.display import Image                 
from IPython.display import display               
from time import gmtime, strftime                 
from sagemaker.predictor import csv_serializer

#test dataset
from sklearn.datasets import load_iris
import pandas as pd   

# Define IAM role
role = get_execution_role()
prefix = 'sagemaker/DEMO-xgboost-dm'
containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'} # each region has its XGBoost container
my_region = boto3.session.Session().region_name # set the region of the instance
print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + containers[my_region] + " container for your SageMaker endpoint.")


# Step 2 
bucket_name = 'Musetest1' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
s3 = boto3.resource('s3')
try:
    if  my_region == 'us-east-1':
      s3.create_bucket(Bucket=bucket_name)
    else: 
      s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)
    
# Step 3 - Load dataset from Kaggle via kagglehub (configurable)
# You can also download the dataset manually and set KAGGLE_FILE_PATH to the CSV filename.
# Configuration: override via environment variables or change defaults here
KAGGLE_DATASET_SLUG = os.environ.get('KAGGLE_DATASET_SLUG', 'muhammadatefelkaffas/eeg-muse2-motor-imagery-brain-electrical-activity')
KAGGLE_FILE_PATH = os.environ.get('KAGGLE_FILE_PATH', '')
LABEL_COL = os.environ.get('LABEL_COL', None)
# If SKIP_LOCAL_PREPROCESS is True the script will download dataset to disk and upload raw CSV to S3
SKIP_LOCAL_PREPROCESS = os.environ.get('SKIP_LOCAL_PREPROCESS', 'True').lower() in ['1','true','yes']
LOCAL_CSV_PATH = os.environ.get('LOCAL_CSV_PATH', 'downloaded_dataset.csv')

def download_dataset_to_disk_via_kaggle_cli(slug, file_path, target_path):
    """Use the Kaggle CLI to download a dataset to disk (unzip into current directory)."""
    # This avoids importing kagglehub and is useful for streaming large files to disk first.
    # Requires 'kaggle' to be installed and configured with kaggle.json credentials.
    import subprocess
    args = ['kaggle', 'datasets', 'download', '-d', slug, '-p', os.path.dirname(target_path) or '.', '--unzip']
    if file_path:
        # if file_path specified, pass it as a file argument after unzip (CLI will still download whole dataset)
        pass
    print('Running Kaggle CLI download:', ' '.join(args))
    subprocess.check_call(args)
    # If dataset produced a single file (by name), ensure it's moved/renamed to target_path
    # Caller should handle locating the downloaded file if multiple files exist.

def load_dataset_via_kagglehub(slug, file_path):
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except Exception as e:
        raise RuntimeError('kagglehub not installed or failed to import: ' + str(e))
    print('Loading Kaggle dataset via kagglehub:', slug)
    return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, slug, file_path)

# If SKIP_LOCAL_PREPROCESS, prefer to download dataset to disk (kaggle CLI) then upload to S3
model_data = None
if SKIP_LOCAL_PREPROCESS:
    print('SKIP_LOCAL_PREPROCESS enabled; attempting to download dataset to disk and upload to S3')
    # First try to download via kaggle CLI (more memory-friendly)
    try:
        download_dataset_to_disk_via_kaggle_cli(KAGGLE_DATASET_SLUG, KAGGLE_FILE_PATH, LOCAL_CSV_PATH)
        # Attempt to locate a CSV in current directory or expected path
        if os.path.exists(LOCAL_CSV_PATH):
            csv_path = LOCAL_CSV_PATH
        else:
            # attempt to find a CSV matching the KAGGLE_FILE_PATH or any csv in cwd
            candidates = [f for f in os.listdir('.') if f.lower().endswith('.csv')]
            if len(candidates) == 1:
                csv_path = candidates[0]
            else:
                # If multiple CSVs, prefer KAGGLE_FILE_PATH name
                csv_path = KAGGLE_FILE_PATH if KAGGLE_FILE_PATH and os.path.exists(KAGGLE_FILE_PATH) else None
        if not csv_path:
            raise FileNotFoundError('Could not find a CSV to upload after Kaggle CLI download. Set LOCAL_CSV_PATH or KAGGLE_FILE_PATH accordingly.')
        print('Uploading', csv_path, 'to S3')
        s3_path = os.path.join(prefix, 'train', 'train.csv')
        boto3.Session().resource('s3').Bucket(bucket_name).Object(s3_path).upload_file(csv_path)
        s3_input_train = sagemaker.inputs.TrainingInput(s3_data=f's3://{bucket_name}/{prefix}/train', content_type='csv')
        print('Uploaded raw CSV to S3 and configured training input. Skipping local train/test split.')
    except Exception as e:
        print('Failed to download/upload via kaggle CLI:', e)
        print('Falling back to loading via kagglehub into memory (may be heavy).')
        model_data = load_dataset_via_kagglehub(KAGGLE_DATASET_SLUG, KAGGLE_FILE_PATH)
else:
    # Load into memory via kagglehub and run preprocessing
    model_data = load_dataset_via_kagglehub(KAGGLE_DATASET_SLUG, KAGGLE_FILE_PATH)

if model_data is not None:
    print('Success: Kaggle dataset loaded. Shape:', getattr(model_data, 'shape', None))
    # Detect label column if not provided
    if LABEL_COL and LABEL_COL in model_data.columns:
        label_col = LABEL_COL
    else:
        label_col = None
        for candidate in ['target', 'label', 'class', 'y', 'y_true', 'gesture', 'event', 'y_yes', 'y_no']:
            if candidate in model_data.columns:
                label_col = candidate
                print('Auto-detected label column:', label_col)
                break
        if label_col is None:
            raise KeyError('Could not find a label column in the dataset. Set LABEL_COL to the correct column name.')

    # If label is string/categorical, try to convert to binary numeric (0/1)
    if model_data[label_col].dtype == object or str(model_data[label_col].dtype).startswith('category'):
        model_data[label_col] = model_data[label_col].map(lambda x: 1 if str(x).lower() in ['yes','y','1','true','t'] else 0)

    # Create y_yes/y_no columns to match the rest of this script if they don't exist
    if 'y_yes' not in model_data.columns:
        model_data['y_yes'] = model_data[label_col]
    if 'y_no' not in model_data.columns:
        model_data['y_no'] = 1 - model_data['y_yes']

    # Basic categorical encoding for non-numeric columns (drop label columns first)
    non_numeric = model_data.drop(['y_no', 'y_yes'], axis=1).select_dtypes(include=['object','category']).columns.tolist()
    if non_numeric:
        print('Found categorical columns, applying one-hot encoding for:', non_numeric)
        dummies = pd.get_dummies(model_data[non_numeric], drop_first=True)
        model_data = pd.concat([model_data.drop(non_numeric, axis=1), dummies], axis=1)

    # Local train/test split and upload
    train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
    print(train_data.shape, test_data.shape)

    # create train.csv expected by XGBoost (label first, no header)
    pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
    boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
    s3_input_train = sagemaker.inputs.TrainingInput(s3_data=f's3://{bucket_name}/{prefix}/train', content_type='csv')

    # test_data is available for evaluation later in the script
else:
    # model_data None means SKIP_LOCAL_PREPROCESS path already set s3_input_train; test_data won't be available
    test_data = None

     
 #step 4
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)

#step 5 
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
#this is where you add training data
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')

#step 6 
sess = sagemaker.Session()
xgb = sagemaker.estimator.Estimator(containers[my_region],role, instance_count=1, instance_type='ml.m4.xlarge',output_path='s3://{}/{}/output'.format(bucket_name, prefix),sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5,eta=0.2,gamma=4,min_child_weight=6,subsample=0.8,silent=0,objective='binary:logistic',num_round=100)

#step 7 
#change this to match
xgb.fit({'train': s3_input_train})

#step 8 
xgb_predictor = xgb.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')

#step 9 
from sagemaker.serializers import CSVSerializer

test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
xgb_predictor.serializer = CSVSerializer() # set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
print(predictions_array.shape)


#step 10 
cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))


# clean up 
#step 11
xgb_predictor.delete_endpoint()
xgb_predictor.delete_model()



# step 12 
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()