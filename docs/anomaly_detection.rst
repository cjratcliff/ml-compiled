""""""""""""""""""""
Anomaly detection
""""""""""""""""""""
This problem can be solved well through methods for density estimation. In addition, the following methods also exist:

Isolation Forest
-------------------
An ensemble of decision trees. The key idea is that points in less dense areas will require fewer splits to be uniquely identified since they are surrounded by fewer points.

Features and split values are randomly chosen, with the split value being somewhere between the min and the max observed values of the feature.

Local Outlier Factor
-----------------------
A nearest-neighbour model.

One-Class SVM
----------------
Learns the equation for the smallest possible hypersphere that totally encapsulates the data points.

