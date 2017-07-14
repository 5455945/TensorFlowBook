import pandas as pd
import tensorflow.contrib.learn as skflow
from sklearn import metrics
from sklearn.model_selection import train_test_split

from data_processing import get_test_data, get_train_data

train_data = get_train_data()
X = train_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Child',
                'EmbarkedF', 'DeckF', 'TitleF', 'Honor']].as_matrix()
Y = train_data['Survived']

# split training data and validation set data
X_train, X_val, Y_train, Y_val = (
    train_test_split(X, Y, test_size=0.1, random_state=42))

# skflow classifier
feature_cols = skflow.infer_real_valued_columns_from_input(X_train)
classifier = skflow.LinearClassifier(feature_columns=feature_cols, n_classes=2)
classifier.fit(X_train, Y_train, steps=200)
score = metrics.accuracy_score(Y_val, classifier.predict(X_val))
print("Accuracy: %f" % score)

# predict on test dataset
test_data = get_test_data()
X = test_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Child',
               'EmbarkedF', 'DeckF', 'TitleF', 'Honor']].as_matrix()
predictions = classifier.predict(X)
submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
})
submission.to_csv("titanic-submission.csv", index=False)
'''
WARNING:tensorflow:float64 is not supported by many models, consider casting to float32.
WARNING:tensorflow:Using temporary folder as model directory: C:\Users\soft\AppData\Local\Temp\tmpc5meq6vi
WARNING:tensorflow:From D:/git/DeepLearning/TensorFlowBook/Titanic/03_skflow.py:20: calling BaseEstimator.fit (from tensorflow.contrib.learn.python.learn.estimators.estimator) with x is deprecated and will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.
Example conversion:
  est = Estimator(...) -> est = SKCompat(Estimator(...))
WARNING:tensorflow:From D:/git/DeepLearning/TensorFlowBook/Titanic/03_skflow.py:20: calling BaseEstimator.fit (from tensorflow.contrib.learn.python.learn.estimators.estimator) with y is deprecated and will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.
Example conversion:
  est = Estimator(...) -> est = SKCompat(Estimator(...))
WARNING:tensorflow:float64 is not supported by many models, consider casting to float32.
WARNING:tensorflow:From C:\Python36\lib\site-packages\tensorflow\contrib\learn\python\learn\estimators\head.py:625: scalar_summary (from tensorflow.python.ops.logging_ops) is deprecated and will be removed after 2016-11-30.
Instructions for updating:
Please switch to tf.summary.scalar. Note that tf.summary.scalar uses the node name instead of the tag. This means that TensorFlow will automatically de-duplicate summary names based on the scope they are created in. Also, passing a tensor or list of tags to a scalar summary op is no longer supported.
WARNING:tensorflow:Casting <dtype: 'int64'> labels to bool.
WARNING:tensorflow:Casting <dtype: 'int64'> labels to bool.
WARNING:tensorflow:From C:\Python36\lib\site-packages\tensorflow\python\util\deprecation.py:347: calling LinearClassifier.predict (from tensorflow.contrib.learn.python.learn.estimators.linear) with outputs=None is deprecated and will be removed after 2017-03-01.
Instructions for updating:
Please switch to predict_classes, or set `outputs` argument.
WARNING:tensorflow:From C:\Python36\lib\site-packages\tensorflow\contrib\learn\python\learn\estimators\linear.py:565: calling BaseEstimator.predict (from tensorflow.contrib.learn.python.learn.estimators.estimator) with x is deprecated and will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.
Example conversion:
  est = Estimator(...) -> est = SKCompat(Estimator(...))
WARNING:tensorflow:float64 is not supported by many models, consider casting to float32.
2017-06-30 17:57:09.918373: I d:\git\deeplearning\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Quadro M2000M, pci bus id: 0000:01:00.0)
Traceback (most recent call last):
  File "D:/git/DeepLearning/TensorFlowBook/Titanic/03_skflow.py", line 21, in <module>
    score = metrics.accuracy_score(Y_val, classifier.predict(X_val))
  File "C:\Python36\lib\site-packages\sklearn\metrics\classification.py", line 172, in accuracy_score
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
  File "C:\Python36\lib\site-packages\sklearn\metrics\classification.py", line 72, in _check_targets
    check_consistent_length(y_true, y_pred)
  File "C:\Python36\lib\site-packages\sklearn\utils\validation.py", line 177, in check_consistent_length
    lengths = [_num_samples(X) for X in arrays if X is not None]
  File "C:\Python36\lib\site-packages\sklearn\utils\validation.py", line 177, in <listcomp>
    lengths = [_num_samples(X) for X in arrays if X is not None]
  File "C:\Python36\lib\site-packages\sklearn\utils\validation.py", line 122, in _num_samples
    type(x))
TypeError: Expected sequence or array-like, got <class 'generator'>
'''