X_test = readtable('X_test2.csv');
X_train = readtable('X_train2.csv');
y_test = readtable('y_test2.csv');
y_train = readtable('y_train2.csv');
X_test = table2array(X_test);
X_train = table2array(X_train);
y_test = table2array(y_test);
y_train = table2array(y_train);

X_test = transpose(X_test);
X_train = transpose(X_train);
y_test = transpose(y_test);
y_train = transpose(y_train);

