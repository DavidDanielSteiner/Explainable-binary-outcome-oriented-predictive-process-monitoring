# -*- coding: utf-8 -*-
"""
Created on Mon Sep 7 15:07:58 2021

@author: David Steiner
"""

#!pip install imblearn
#!pip install pm4py

import pandas as pd
from pandas import Series
import numpy as np
from numpy import arange

import seaborn as sns
import matplotlib.pyplot as plt

import gzip
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# =============================================================================
# Settings
# =============================================================================

case_id_col = activity_col =  timestamp_col =  label_col =  resource_col = 'empty'
event_categorical_attributes = event_numeric_attributes = case_categorical_attributes = case_numeric_attributes = 'empty'
static_cols = dynamic_cols = cat_cols= 'empty'
dt_first_last_timestamps = 'empty'
 
case_id_col = "case:concept:name"   # Case ID
activity_col = "concept:name"       # Event Name
timestamp_col = 'time:timestamp'    
label_col = 'label'                 # Output Label
resource_col = 'org:resource'    


def get_dataset_settings(dataset_name):
    """
    Imports dataset and specifies feature categories. 
    Optionally makes minor data adjustments.
    Returns preprocessed dataset and list of feature categories.
    """

    global case_id_col, activity_col, timestamp_col, label_col, resource_col
    global event_categorical_attributes, event_numeric_attributes, case_categorical_attributes, case_numeric_attributes
    global static_cols, dynamic_cols, cat_cols
    
    case_id_col = "case:concept:name"   # Case ID
    activity_col = "concept:name"       # Event Name
    timestamp_col = 'time:timestamp'    
    label_col = 'label'                 # Output Label
    resource_col = 'org:resource'       # specific attribute (not required)
    
    
    if dataset_name == 'BPIC17':
        data = pd.read_csv("BPIC_2017.csv", index_col = False)
        
        data.loc[data['CreditScore']=='missing', 'CreditScore'] = -5
        data['CreditScore'] = pd.to_numeric(data['CreditScore'])
        
    if dataset_name == 'BPIC13': 
        data = pd.read_csv("BPIC_2013.csv", index_col = False)
        
    if dataset_name == 'BPIC19':      
        data = pd.read_csv("BPIC_2019.csv", index_col = False)
        data = data.drop(columns=['User'])
        data = data.rename(columns={'Cumulative net worth (EUR)':'Cumulative net worth', 
                                    'case:Purch. Doc. Category name': 'case:PurchDocCategoryname',
                                    'case:GR-Based Inv. Verif.' : 'case:GR-BasedInvVerif'})
        
    if dataset_name == 'BPIC15':      
        data = pd.read_csv("BPIC_15.csv", index_col = False)
        
        data = data.drop(columns=['case:landRegisterID', 'dateFinished', 'dueDate', 'action_code', 'planned', 'activityNameNL', 'concept:name', 'case:endDate', 'case:endDatePlanned', 'case:startDate'])
        data = data.rename(columns={"activityNameEN": "concept:name"})       
        
    if dataset_name == 'BPICSepsis':      
        data = pd.read_csv("BPI_Sepsis.csv", index_col = False)
        
    if dataset_name == 'BPICHospital':      
        data = pd.read_csv("BPI_HospitalBilling.csv", index_col = False)
        
        # remove incomplete cases
        tmp = data.groupby(case_id_col).apply(check_if_any_of_activities_exist, activities=["BILLED", "DELETE", "FIN"])
        incomplete_cases = tmp.index[tmp==False]
        data = data[~data[case_id_col].isin(incomplete_cases)]


    agg_unique = data.groupby(case_id_col).nunique().aggregate(pd.Series.nunique)

    static_cols = agg_unique[agg_unique == 1].keys()
    dynamic_cols = agg_unique[agg_unique != 1].keys()
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()

    if case_id_col in cat_cols:
        cat_cols.remove(case_id_col)
    if label_col in cat_cols:
        cat_cols.remove(label_col)
    if activity_col in cat_cols:
        cat_cols.remove(activity_col)
    if timestamp_col in cat_cols:
        cat_cols.remove(timestamp_col)

    static_cols = data[data.columns.intersection(static_cols)]
    case_numeric_attributes = static_cols.select_dtypes(include=np.number).columns.tolist()
    case_categorical_attributes = static_cols.select_dtypes(include=['object']).columns.tolist()
    case_categorical_attributes.remove(label_col)

    dynamic_cols = data[data.columns.intersection(dynamic_cols)]
    event_numeric_attributes = dynamic_cols.select_dtypes(include=np.number).columns.tolist()
    event_categorical_attributes = dynamic_cols.select_dtypes(include=['object']).columns.tolist()
    event_categorical_attributes.remove(timestamp_col)

    print('Categoric Event Attributes:', len(event_categorical_attributes), (event_categorical_attributes), '\n')
    print('Numeric Event Attributes:', len(event_numeric_attributes), (event_numeric_attributes), '\n')
    print('Categoric Case Attributes:', len(case_categorical_attributes), (case_categorical_attributes), '\n')
    print('Numeric Case Attributes:', len(case_numeric_attributes), (case_numeric_attributes), '\n')

    print('Dataset Shape',data.shape)
        
    return data, case_id_col, activity_col, timestamp_col, label_col, resource_col, event_categorical_attributes, event_numeric_attributes, case_categorical_attributes, case_numeric_attributes, static_cols, dynamic_cols, cat_cols


# =============================================================================
# Data Preprocessing
# =============================================================================

def xes_converter(xes_file_name):
    """
    Converts xes format to tabular format.
    Optionally unzips eventlog.
    """
    
    try:
        inFile = gzip.open(xes_file_name, 'rb')
        outFile = open('eventlog.xes','wb')
        outFile.write(inFile.read())
        inFile.close()
        outFile.close()
        xes_log = xes_importer.apply('eventlog.xes')
    except:  
        xes_log = xes_importer.apply(xes_file_name)
        
    data = log_converter.to_data_frame.apply(xes_log)
    return data

# =============================================================================
# Feature engineering
# =============================================================================

def extract_timestamp_features(group):
    """
    Helper Function to extract features "timesincelastevent" and "event_nr" 
    from the timestamp.
    Source: https://github.com/irhete/predictive-monitoring-benchmark
    """
 
    group = group.sort_values(timestamp_col, ascending=False, kind='mergesort')
    
    tmp = group[timestamp_col] - group[timestamp_col].shift(-1)
    tmp = tmp.fillna(pd.Timedelta(seconds=0))
    group["timesincelastevent"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) # m is for minutes

    tmp = group[timestamp_col] - group[timestamp_col].iloc[-1]
    tmp = tmp.fillna(pd.Timedelta(seconds=0))
    group["timesincecasestart"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) # m is for minutes

    group = group.sort_values(timestamp_col, ascending=True, kind='mergesort')
    group["event_nr"] = range(1, len(group) + 1)
    
    return group


def get_open_cases(date):
    """
    Helper function to get number of open cases
    Source: https://github.com/irhete/predictive-monitoring-benchmark
    """
    return sum((dt_first_last_timestamps["start_time"] <= date) & (dt_first_last_timestamps["end_time"] > date))

def add_timestap_features(data):
    """
    Extracts generic features from the timestamp
    Source: https://github.com/irhete/predictive-monitoring-benchmark
    """


    print("Extracting timestamp features")
    try:
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], utc=True)
    except:
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data["timesincemidnight"] = data[timestamp_col].dt.hour * 60 + data[timestamp_col].dt.minute
    data["month"] = data[timestamp_col].dt.month
    data["weekday"] = data[timestamp_col].dt.weekday
    data["hour"] = data[timestamp_col].dt.hour

    # add features extracted from timestamp
    data = data.groupby(case_id_col).apply(extract_timestamp_features)
    return data

def add_inter_case_features(data):
    """
    Extracts the number of open cases at the time of each event from the timestamp
    Source: https://github.com/irhete/predictive-monitoring-benchmark
    """

    global dt_first_last_timestamps
    
    print("Extracting open cases")
    data = data.sort_values([timestamp_col], ascending=True, kind='mergesort')
    dt_first_last_timestamps = data.groupby(case_id_col)[timestamp_col].agg([min, max])
    dt_first_last_timestamps.columns = ["start_time", "end_time"]
    data["open_cases"] = data[timestamp_col].apply(get_open_cases)
    return data

def impute_missing_values(data):
    """
    Imputes missing values for numerical data and fills 'missing' for missing 
    categorical data
    Source: https://github.com/irhete/predictive-monitoring-benchmark
    """
    
    print('Imputing missing values')
    data[label_col] = 'empty' #placeholder
    grouped = data.sort_values(timestamp_col, ascending=True, kind='mergesort').groupby(case_id_col)
    
    for col in list(data.columns.values):
        data[col] = grouped[col].transform(lambda grp: grp.fillna(method='ffill'))

    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    data[cat_cols] = data[cat_cols].fillna('missing')
    data = data.fillna(0)
    
    #filter and sort
    data = data[data.columns.values]
    data = data.sort_values([case_id_col, timestamp_col], ascending = True)
    return data



# =============================================================================
# Trace Bucketing
# =============================================================================

def create_trace_bucket(data, cut_after_events=10, get_case_progress=False):
    """
    Creates a bucket with cases up to a specified prefix length
    """
    
    trace_bucket = data[data['event_nr'] <= cut_after_events]

    last_time = data.groupby(case_id_col)['timesincecasestart'].agg('last')
    avg_total_time = last_time.mean()
    last_time = trace_bucket.groupby(case_id_col)['timesincecasestart'].agg('last')
    avg_trace_bucket_time = last_time.mean()
    time_prediction = avg_total_time - avg_trace_bucket_time 
    
    print("Making predictions after", cut_after_events, 'completed events')
    print('Making predictions at time after cases have started on average: ', round(avg_trace_bucket_time / 60, 1), 'hours/', round(avg_trace_bucket_time / 60 / 24, 3) , 'days')
    print('Making predictions at time before cases have finished on average: ', round(time_prediction / 60, 1), 'hours/', round(time_prediction / 60 / 24, 1) , 'days')
        
    print('\n Shape:', trace_bucket.shape)
    
    if get_case_progress:
        avg_case_progress = (avg_trace_bucket_time / avg_total_time) * 100
        return trace_bucket, avg_case_progress
    else:
        return trace_bucket



def create_mixed_bucket(data, min_event_length, max_event_length):
    """
    Experimental setup
    Creates a single bucket with different trace lengths
    """

    import random

    ids = data[case_id_col].unique()
    print(len(ids))
    random.shuffle(ids)
    id_split_list = np.array_split(ids, max_event_length+1-min_event_length)

    all_buckets = pd.DataFrame()

    for i in range(min_event_length,max_event_length+1):
        print(i)
        tmp = data[data[case_id_col].isin(id_split_list[i-min_event_length])]
        trace_bucket = create_trace_bucket(tmp, i)
        all_buckets = all_buckets.append(trace_bucket)


# =============================================================================
# 
# =============================================================================

def remove_features(trace_bucket, event_categorical_attributes, case_categorical_attributes, event_numeric_attributes, case_numeric_attributes, only_events=False):
    """
    Removes selected attributes from the trace bucket
    """
    static_cols = case_categorical_attributes + case_numeric_attributes + [case_id_col, label_col]
    dynamic_cols = event_categorical_attributes + event_numeric_attributes + [timestamp_col]
    cat_cols = event_categorical_attributes + case_categorical_attributes
        
    n_features_before = len(trace_bucket.columns)
    trace_bucket = trace_bucket[static_cols + dynamic_cols]
    n_features_after = len(trace_bucket.columns)
    print('Features removed: ', n_features_before-n_features_after)
     
    return trace_bucket


def remove_events(data, trace_bucket, drop_events_list):
    """
    Remove selected events from trace bucket
    """
    trace_bucket = trace_bucket[~data[activity_col].isin(drop_events_list)]
    
    print('\n Events included:')
    print(trace_bucket[activity_col].unique())   
    
    return trace_bucket


def split_data_temporal(data, train_ratio):  
    """
    Splitting the data into train and test data based on the time the case finished
    Source: https://github.com/irhete/predictive-monitoring-benchmark
    """
    
    sorting_cols = [timestamp_col, activity_col]
    
    # split into train and test using temporal split and discard events that overlap the periods
    data = data.sort_values(sorting_cols, ascending=True, kind='mergesort')
    grouped = data.groupby(case_id_col)
    start_timestamps = grouped[timestamp_col].min().reset_index()
    start_timestamps = start_timestamps.sort_values(timestamp_col, ascending=True, kind='mergesort')
    train_ids = list(start_timestamps[case_id_col])[:int(train_ratio*len(start_timestamps))]
    train = data[data[case_id_col].isin(train_ids)].sort_values(sorting_cols, ascending=True, kind='mergesort')
    test = data[~data[case_id_col].isin(train_ids)].sort_values(sorting_cols, ascending=True, kind='mergesort')
    split_ts = test[timestamp_col].min()
    train = train[train[timestamp_col] < split_ts]
    train.drop([timestamp_col],axis=1,inplace=True)
    test.drop([timestamp_col],axis=1,inplace=True)
        
    return train, test


def replace_missing_cols(train, test):
    """
    Created additional empty feature columns if categorical feature levels are not present
    in the test or the train dataset.
    """
    missing_col_train = test.columns.difference(train.columns).tolist()
    missing_col_test = train.columns.difference(test.columns).tolist()
    print(missing_col_train, missing_col_train)
    train[missing_col_train] = 0
    header_names = test.columns.tolist()
    train = train[header_names]
    test[missing_col_test] = 0
    header_names = train.columns.tolist()
    test = test[header_names]
    
    return train, test


def prepare_ml_train_test(train,test, balanced=False):
    """
    Preparing the train and test data for classical machine learning models.
    Optionally balances the dataset by random undersampling the majority class.
    """
    try:
        train = train.drop(columns=['Event_Sequence'])
        test = test.drop(columns=['Event_Sequence'])
    except:
        pass
    
    X_train = train.drop(columns=[label_col])
    X_test = test.drop(columns=[label_col])
    y_train = train[[label_col]]
    y_test = test[[label_col]]
    
    if balanced==True:
        print('Undersampling Train Data')
        ros = RandomUnderSampler()
        X_train, y_train = ros.fit_resample(X_train, y_train)
    
    print(X_train.shape, X_test.shape)
    return X_train, y_train, X_test, y_test


def group_infrequent_features(data, max_category_levels = 15):
    """
    Grouping infrequent factor levels that are not among the top n most common
    """

    print('Binning and Grouping infrequent factor levels to: Other_infrequent')
    for col in cat_cols:

        #Get only most frequent category levels and group all others
        if col != activity_col:
            counts = data[col].value_counts()
            mask = data[col].isin(counts.index[max_category_levels:])
            data.loc[mask, col] = "Other_infrequent"
            
    return data 


# =============================================================================
# Sequence Encoding
# =============================================================================

def aggregate_data(data: pd.DataFrame,                
                   case_id_col: str,
                   activity_col: str,
                   label_col: str,
                   case_numeric_attributes: list,
                   case_categorical_attributes: list,
                   event_numeric_attributes: list,
                   event_categorical_attributes: list,
                    d_event_sequence = True,
                    d_event = True,
                    d_event_categorical = True,
                    d_case_categorical =True,
                    one_hot_case_categorical = True,   
                    d_event_numeric = True,
                    d_case_numeric = True):
    
    """
    Consolidates each case containing multiple rows into one row. 
    Last State based encoding for case attributes.
    Max aggregation for numerical event attributes.
    Optionally One Hot encoding for categorical event attributes.
    """
        
        
    data_agg = data[[case_id_col, label_col]].drop_duplicates(subset=case_id_col, keep='first', inplace=False, ignore_index=False).set_index(case_id_col)
    
    ### Aggregate Categorical

    #one hot encode events/activities
    #data = get_dummies(data, columns=[activity_col], prefix="event")
    if d_event:
        tmp = event_categorical_attributes.copy()
        tmp.append(case_id_col)
        df_event_categorical = data[tmp]
        df_event_categorical = df_event_categorical[[case_id_col, activity_col]]

        for col in df_event_categorical:
            if col != case_id_col:
                #print("One hot encode", col)
                df_event_categorical = pd.get_dummies(df_event_categorical, columns=[col], prefix=col)

        df_activity = df_event_categorical.groupby(case_id_col).agg(max)      
        data_agg = data_agg.join(df_activity)
        
    
    if d_event_categorical:
        #aggregate and one hot encode categorical event attributes
        tmp = event_categorical_attributes.copy()
        tmp.append(case_id_col)
        tmp.remove(activity_col)
        df_event_categorical = data[tmp]

        for col in df_event_categorical:
            if col != case_id_col:
                #print("One hot encode", col)
                df_event_categorical = pd.get_dummies(df_event_categorical, columns=[col], prefix=col)

        df_event_categorical = df_event_categorical.groupby(case_id_col).agg(max)
        data_agg = data_agg.join(df_event_categorical)
        

    if d_case_categorical:
        #aggregate (dummy encode) case attributes  
        tmp = case_categorical_attributes.copy()
        tmp.append(case_id_col)
        df_case_categorical = data[tmp]
            
        if one_hot_case_categorical:
            for col in case_categorical_attributes:
                if col != case_id_col:
                    #print("One hot encode", col)
                    df_case_categorical = pd.get_dummies(df_case_categorical, columns=[col], prefix=col)

            df_case_categorical = df_case_categorical.groupby(case_id_col).agg(max)
            data_agg = data_agg.join(df_case_categorical)
        
        else:
            #Not one hot encode                
            df_case_categorical = df_case_categorical.groupby(case_id_col).agg('first')
            data_agg = data_agg.join(df_case_categorical) 
        

    ### Aggregate Numerical
    
    if d_case_numeric:
        #aggregate numeric case attributes
        tmp = case_numeric_attributes.copy()
        df_case_numeric = data.groupby(case_id_col)[tmp].agg('first')
        data_agg = data_agg.join(df_case_numeric)

    if d_event_numeric:
        #aggregatet numeric event attributes  
        #tmp_mean = data.groupby(case_id_col)[event_numeric_attributes].agg('mean')
        #tmp_mean.columns = [str(col) + '_mean' for col in tmp_mean.columns]

        tmp_max = data.groupby(case_id_col)[event_numeric_attributes].agg('max')
        tmp_max.columns = [str(col) + '_max' for col in tmp_max.columns]

        #tmp_mode = data.groupby(case_id_col)[event_numeric_attributes].agg(lambda x: pd.Series.mode(x)[0])
        #tmp_mode.columns = [str(col) + '_mode' for col in tmp_mode.columns]

        df_event_numeric_attributes = tmp_max
        #df_event_numeric_attributes = tmp_mean.join(tmp_max)
        data_agg = data_agg.join(df_event_numeric_attributes)
        #df_event_numeric_attributes = df_event_numeric_attributes.join(tmp_mode)
    
      
    if d_event_sequence:    
        #add sequence of events
        df = data.copy()
        remove_characters  = [' ', '_', '(', ')']
        for character in remove_characters:
            df[activity_col] = df[activity_col].str.replace(character, "")
        df['Event_Sequence'] = df.groupby([case_id_col])[activity_col].transform(lambda x: ' '.join(x))
        df = df.reset_index(drop=True)
        df['Event_Sequence'][1]

        df_event_sequence = df.groupby(case_id_col)['Event_Sequence'].agg('first')
        data_agg = data_agg.join(df_event_sequence)
    
   
    return data_agg


# =============================================================================
# Deep Learning Preprocessing
# =============================================================================

def scale_data(train, test):
    """
    Scale all numerical columns to zero mean and unit variance
    """
    
    num_cols = train.select_dtypes(include=np.number).columns.tolist()
    num_cols.remove(label_col)

    for col in num_cols: 
        scale = StandardScaler().fit(train[[col]])   
        train[col] = scale.transform(train[[col]])
        test[col] = scale.transform(test[[col]])
        
    return train, test

def one_hot_encode(event_log: pd.DataFrame):
    """
    One-hot encode all categorical features. 
    """
        
    cat_cols = event_log.select_dtypes(include=['object']).columns.tolist()
    cat_cols.remove(case_id_col)
    
    oh_encoded = pd.get_dummies(data=event_log, columns = cat_cols)
                                   
    return oh_encoded


def prepare_dl_train_test(oh_encoded, max_sequence_length, balanced=False):
    """
    Transform the input dataframe to a 3D array containing
    sequences of max_sequence_length
    Optionally balance train data by random undersampling the majority class
    """
    
    sequences = oh_encoded.groupby(case_id_col).agg(lambda x: list(x))

    #if balanced==True:
    #    print('Undersampling Train Data')
    #    ros = RandomUnderSampler()
    #    X_train, y_train = ros.fit_resample(X_train, y_train) 
    #    print(X_train.shape, X_test.shape)
    
    #Label 
    y = sequences[[label_col]].to_numpy().ravel()
    y = np.array([item[0] for item in y])
    
    #Features
    sequences = sequences.drop(columns=[label_col])
    
    feature_names = sequences.columns.values

    prepared_features = []

    for column in sequences.columns:
        feature = np.array([np.array(xi) for xi in sequences[column]], dtype=object)
        feature = pad_sequences(feature, maxlen=max_sequence_length, dtype='float')
        feature = np.expand_dims(feature, axis=2)
        prepared_features.append(feature)

    feature_vector = np.concatenate(prepared_features, -1)

    return feature_vector, y, feature_names



# =============================================================================
# Outcome Labeling Functions
# =============================================================================

label_col = "label"
pos_label = 1
neg_label = 0

def cut_trace_before_activity(group, relevant_activity):
    """
    Cut cases before relevant activity happens
    Source: https://github.com/irhete/predictive-monitoring-benchmark
    """
    relevant_activity_idxs = np.where(group[activity_col] == relevant_activity)[0]
    if len(relevant_activity_idxs) > 0:
        cut_idx = relevant_activity_idxs[0]
        return group[:cut_idx]
    else:
        return group


def check_if_any_of_activities_exist(group, activities):
    """
    Cut cases before any relevant activity happens
    Source: https://github.com/irhete/predictive-monitoring-benchmark
    """
    if np.sum(group[activity_col].isin(activities)) > 0:
        return True
    else:
        return False
    

def check_if_activity_exists(group, activity, cut_from_idx=True):
    """
    Helper function to find out if activicty exists
    Source: https://github.com/irhete/predictive-monitoring-benchmark
    """
    relevant_activity_idxs = np.where(group[activity_col] == activity)[0]
    if len(relevant_activity_idxs) > 0:
        idx = relevant_activity_idxs[0]
        group[label_col] = pos_label
        if cut_from_idx:
            return group[:idx]
        else:
            return group
    else:
        group[label_col] = neg_label
        return group   

def check_if_attribute_exists(group, attribute, cut_from_idx=True):
    """
    Helper function to find out if attribute exists
    Source: https://github.com/irhete/predictive-monitoring-benchmark
    """
    group[label_col] = neg_label if True in list(group[attribute]) else pos_label
    relevant_idxs = np.where(group[attribute]==True)[0]
    if len(relevant_idxs) > 0:
        cut_idx = relevant_idxs[0]
        if cut_from_idx:
            return group[:idx]
        else:
            return group
    else:
        return group


# =============================================================================
# Outcome Labeling
# =============================================================================

def define_binary_outcome_label(data, attributes, outcome_label):
    """
    Labeling the dataset based on pre-specified labeling functions
    """
    
    event_categorical_attributes = attributes[0].copy()
    case_categorical_attributes = attributes[1].copy()
    event_numeric_attributes = attributes[2].copy()
    case_numeric_attributes = attributes[3].copy()
    
    
    if outcome_label == 'BPIC17-LoanAccepted':
        # Accepted or not accepted
        relevant_offer_events = ["O_Cancelled", "O_Accepted", "O_Refused"]
        activity='O_Accepted'

        print("Assigning class labels...")
        last_o_events = data[data.EventOrigin == "Offer"].sort_values(timestamp_col, ascending=True, kind='mergesort').groupby(case_id_col).last()[activity_col]
        last_o_events = pd.DataFrame(last_o_events)
        last_o_events.columns = ["last_o_activity"]
        data = data.merge(last_o_events, left_on=case_id_col, right_index=True)
        data = data[data.last_o_activity.isin(relevant_offer_events)]

        print("Set labels to 1 for Outcome:", activity)
        data[label_col] = 0
        data.loc[data["last_o_activity"] == activity, label_col] = 1
        #data = data[static_cols + dynamic_cols]

        balance = data.groupby([case_id_col])[label_col].agg('max').reset_index()
        balance = balance.groupby([label_col])[case_id_col].count()
        print(balance)
        
        #Remove every event after Last Acticity
        data = data.sort_values(timestamp_col, kind='mergesort').reset_index(drop=True).groupby(case_id_col).apply(cut_trace_before_activity, ("O_Cancelled"))
        data = data.sort_values(timestamp_col, kind='mergesort').reset_index(drop=True).groupby(case_id_col).apply(cut_trace_before_activity, ("O_Accepted"))
        data = data.sort_values(timestamp_col, kind='mergesort').reset_index(drop=True).groupby(case_id_col).apply(cut_trace_before_activity, ("O_Refused"))

        data = data.reset_index(drop=True)
        
        
        drop_events_list=[]
        event_categorical_attributes.remove('Selected')
        event_categorical_attributes.remove('Accepted')
        
        
    elif outcome_label == 'BPIC17-PotentialFraud': 
    
        fraud_cases = data[data[activity_col] == 'W_Assess potential fraud']
        fraud_cases = list(fraud_cases[case_id_col])

        print("Set labels to 1 for Outcome:", "Asses Potential Fraud")
        data[label_col] = 0
        data.loc[data[case_id_col].isin(fraud_cases), label_col] = 1

        drop_events_list=['W_Assess potential fraud']
        event_categorical_attributes.remove('Selected')

        balance = data.groupby([case_id_col])[label_col].agg('max').reset_index()
        balance = balance.groupby([label_col])[case_id_col].count()
        print(balance)
        
        
        
    elif outcome_label == 'BPIC17-LongRunningCases':    
        
        long_cases = data[data['timesincecasestart'] >= 45000]
        long_cases = list(long_cases[case_id_col])

        print("Set labels to 1 for Outcome:", "Long Running Case")
        data[label_col] = 0
        data.loc[data[case_id_col].isin(long_cases), label_col] = 1

        drop_events_list=[]
        event_numeric_attributes.remove('timesincecasestart')

        balance = data.groupby([case_id_col])[label_col].agg('max').reset_index()
        balance = balance.groupby([label_col])[case_id_col].count()
        print(balance)
        
        
    elif outcome_label == 'BPIC13-SupportLevel-1':   
        
        print("Set labels to 1 for Outcome:", "Resolved in Support Level 1")
        data[label_col] = 1
        data.loc[data['org:group'].str.contains('2nd'), label_col] = 2
        data.loc[data['org:group'].str.contains('3rd'), label_col] = 3

        third_level = data[data[label_col] == 3]
        third_level = list(third_level[case_id_col])
        data.loc[data[case_id_col].isin(third_level), label_col] = 0

        second_level = data[data[label_col] == 2]
        second_level = list(second_level[case_id_col])
        data.loc[data[case_id_col].isin(second_level), label_col] = 0
        
        drop_events_list=[]
        event_categorical_attributes.remove('org:group')
        event_categorical_attributes.remove('org:role')
        event_categorical_attributes.remove('organization involved')
        
        balance = data.groupby([case_id_col])[label_col].agg('max').reset_index()
        balance = balance.groupby([label_col])[case_id_col].count()
        print(balance)
        
    
    elif outcome_label == 'BPIC19-DeletedPO':  
        
        del_po_cases = data[data[activity_col] == 'Delete Purchase Order Item']
        del_po_cases = list(del_po_cases[case_id_col])
        
        print("Set labels to 1 for Outcome:", "Purchase Order Item deleted")
        data[label_col] = 0
        data.loc[data[case_id_col].isin(del_po_cases), label_col] = 1

        #dt_labeled = data[data[activity_col] != 'Delete Purchase Order Item']
        drop_events_list=['Delete Purchase Order Item']
          
        balance = data.groupby([case_id_col])[label_col].agg('max').reset_index()
        balance = balance.groupby([label_col])[case_id_col].count()
        print(balance)
        
        
    elif outcome_label == 'BPICHospital-BillingClosed':      
        
        print("Set labels to 1 for Outcome:", "Billing not closed")
        data = data.groupby(case_id_col).apply(check_if_attribute_exists, attribute="isClosed", cut_from_idx=False)
        
        drop_events_list=[]
        data = data.drop(['isClosed'], axis=1)
        
        balance = data.groupby([case_id_col])[label_col].agg('max').reset_index()
        balance = balance.groupby([label_col])[case_id_col].count()
        print(balance)
        
    
    elif outcome_label == 'BPICHospital-CaseReopened':  
        
        print("Set labels to 1 for Outcome:", "Case will get reopened")
        data = data.groupby(case_id_col).apply(check_if_activity_exists, activity="REOPEN", cut_from_idx=True)
        data = data.reset_index(drop=True)
        
        drop_events_list=[]

        balance = data.groupby([case_id_col])[label_col].agg('max').reset_index()
        balance = balance.groupby([label_col])[case_id_col].count()
        print(balance)
        
    else:
        print('This Outcome Label is not defined')

        
    dl_attributes = [event_categorical_attributes, case_categorical_attributes, event_numeric_attributes, case_numeric_attributes]    
              
    return data, drop_events_list, dl_attributes      








