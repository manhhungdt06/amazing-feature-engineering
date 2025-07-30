# Generated from: day_6.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures


download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()


timeseries.head()


import matplotlib.pyplot as plt
timeseries[timeseries['id'] == 3].plot(subplots=True, sharex=True, figsize=(10,10))
plt.show()



timeseries[timeseries['id'] == 20].plot(subplots=True, sharex=True, figsize=(10,10))
plt.show()


from tsfresh import extract_features
extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
extracted_features


from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

impute(extracted_features)
features_filtered = select_features(extracted_features, y)
features_filtered


from tsfresh import extract_relevant_features

features_filtered_direct = extract_relevant_features(timeseries, y, column_id='id', column_sort='time')
features_filtered_direct

