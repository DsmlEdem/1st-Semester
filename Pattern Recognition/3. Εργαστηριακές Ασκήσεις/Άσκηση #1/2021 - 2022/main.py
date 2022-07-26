import lib # This imports the lib auxiliary file where most functions are defined
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# WARNING! The data files must be present into the same folder as the main.py and lib.py files.
# Load the data into pandas dataframes in case tabular visualizations are needed in the future
traindf = pd.read_csv('train.txt', sep=' ', header=None)
testdf = pd.read_csv('test.txt', sep=' ', header=None)

# Some very basic wrangling
traindf.rename(columns={0 : 'Digit'}, inplace=True)
for i in range(traindf.shape[1]-1):
    traindf.rename(columns={i+1 : 'Pixel '+str(i+1)}, inplace=True)
    
testdf.rename(columns={0 : 'Digit'}, inplace=True)
for i in range(testdf.shape[1]-1):
    testdf.rename(columns={i+1 : 'Pixel '+str(i+1)}, inplace=True)

traindf.drop('Pixel 257', axis=1, inplace=True)
testdf.drop('Pixel 257', axis=1, inplace=True)

traindf['Digit'] = traindf['Digit'].astype(int)
testdf['Digit'] = testdf['Digit'].astype(int)

# Load the required data into numpy arrays
y_train = traindf['Digit'].to_numpy()
X_train = traindf.iloc[:,1:].values
y_test = testdf['Digit'].to_numpy()
X_test = testdf.iloc[:,1:].values
print('Step 1 complete, all data have been loaded into numpy arrays.')

print('Step 2: This is the 131st digit of our training data list.')
lib.show_sample(X_train, 131)

print('Step 3: We shall now print one random sample for each digit.')
lib.plot_digits_samples(X_train, y_train)

print('Steps 4 and 5:')
mean_zero = lib.digit_mean_at_pixel(X_train, y_train, 0, pixel=(10, 10))
var_zero = lib.digit_variance_at_pixel(X_train, y_train, 0, pixel=(10, 10))
print('The mean value of the (10,10) pixel for Digit 0 is {}, while the corresponding variance is {}.'.format(mean_zero, var_zero))

print('Step 6: Let us now calculate the mean value and variance for all pixels of the Digit 0.')
means_zero = lib.digit_mean(X_train, y_train, 0)
vars_zero = lib.digit_variance(X_train, y_train, 0)
print('Step 6 completed.')
print('Step 7: Using the previous mean values, Digit 0 can be represented as:')
plt.imshow(means_zero.reshape(16,16), cmap='gray')
plt.show()

print('Step 8: Let us now plot the same digit, using the previously calculated variance values.')
print('The variance itself can be seen in this figure:')
plt.imshow(vars_zero.reshape(16,16), cmap='gray')
plt.show()
print('As expected, the area where the mean is the highest corresponds to a lower variance. On the other hand, the area around it has a high variance.')

stds_zero = np.sqrt(vars_zero)

print('Below you can see the image corresponding to the mean value of 0, as well as 1 and 2 stds away from it.')
fig = plt.figure(figsize=(15, 5)) # We need 10 subplots, one for each digit
    
ax1 = fig.add_subplot(1,5,1)
sc1 = ax1.imshow((means_zero-2*stds_zero).reshape(16,16), cmap='gray')
ax1.set_title('μ(0) - 2σ')
ax1.set_xticks([0,5,10,15])
ax1.set_yticks([0,5,10,15])
    
ax2 = fig.add_subplot(1,5,2)
sc2 = ax2.imshow((means_zero-stds_zero).reshape(16,16), cmap='gray')
ax2.set_title('μ(0) - σ')
ax2.set_xticks([0,5,10,15])
ax2.set_yticks([0,5,10,15])
    
ax3 = fig.add_subplot(1,5,3)
sc3 = ax3.imshow(means_zero.reshape(16,16), cmap='gray')
ax3.set_title('μ(0)')
ax3.set_xticks([0,5,10,15])
ax3.set_yticks([0,5,10,15])
    
ax4 = fig.add_subplot(1,5,4)
sc4 = ax4.imshow((means_zero+stds_zero).reshape(16,16), cmap='gray')
ax4.set_title('μ(0) + σ')
ax4.set_xticks([0,5,10,15])
ax4.set_yticks([0,5,10,15])
    
ax5 = fig.add_subplot(1,5,5)
sc5 = ax5.imshow((means_zero+2*stds_zero).reshape(16,16), cmap='gray')
ax5.set_title('μ(0) + 2σ')
ax5.set_xticks([0,5,10,15])
ax5.set_yticks([0,5,10,15])
plt.show()

print('Step 9: We shall now perform the same procedure for all digits.')
means = np.empty([10,X_train.shape[1]])
varss = np.empty([10,X_train.shape[1]])

for dig in range(10):
    means[dig] = lib.digit_mean(X_train, y_train, dig)
    varss[dig] = lib.digit_variance(X_train, y_train, dig)

print('In this figure, you can see subplots for all digits, as drawn by their calculated means:')
fig = plt.figure(figsize=(10, 5)) # We need 10 subplots, one for each digit
    
ax1 = fig.add_subplot(2,5,1)
sc1 = ax1.imshow(means[0,:].reshape(16,16), cmap='gray')
ax1.set_title('Digit 0')
ax1.set_xticks([0,5,10,15])
ax1.set_yticks([0,5,10,15])
    
ax2 = fig.add_subplot(2,5,2)
sc2 = ax2.imshow(means[1,:].reshape(16,16), cmap='gray')
ax2.set_title('Digit 1')
ax2.set_xticks([0,5,10,15])
ax2.set_yticks([0,5,10,15])
    
ax3 = fig.add_subplot(2,5,3)
sc3 = ax3.imshow(means[2,:].reshape(16,16), cmap='gray')
ax3.set_title('Digit 2')
ax3.set_xticks([0,5,10,15])
ax3.set_yticks([0,5,10,15])
    
ax4 = fig.add_subplot(2,5,4)
sc4 = ax4.imshow(means[3,:].reshape(16,16), cmap='gray')
ax4.set_title('Digit 3')
ax4.set_xticks([0,5,10,15])
ax4.set_yticks([0,5,10,15])
    
ax5 = fig.add_subplot(2,5,5)
sc5 = ax5.imshow(means[4,:].reshape(16,16), cmap='gray')
ax5.set_title('Digit 4')
ax5.set_xticks([0,5,10,15])
ax5.set_yticks([0,5,10,15])
    
ax6 = fig.add_subplot(2,5,6)
sc6 = ax6.imshow(means[5,:].reshape(16,16), cmap='gray')
ax6.set_title('Digit 5')
ax6.set_xticks([0,5,10,15])
ax6.set_yticks([0,5,10,15])
    
ax7 = fig.add_subplot(2,5,7)
sc7 = ax7.imshow(means[6,:].reshape(16,16), cmap='gray')
ax7.set_title('Digit 6')
ax7.set_xticks([0,5,10,15])
ax7.set_yticks([0,5,10,15])
    
ax8 = fig.add_subplot(2,5,8)
sc8 = ax8.imshow(means[7,:].reshape(16,16), cmap='gray')
ax8.set_title('Digit 7')
ax8.set_xticks([0,5,10,15])
ax8.set_yticks([0,5,10,15])
    
ax9 = fig.add_subplot(2,5,9)
sc9 = ax9.imshow(means[8,:].reshape(16,16), cmap='gray')
ax9.set_title('Digit 8')
ax9.set_xticks([0,5,10,15])
ax9.set_yticks([0,5,10,15])
    
ax10 = fig.add_subplot(2,5,10)
sc10 = ax10.imshow(means[9,:].reshape(16,16), cmap='gray')
ax10.set_title('Digit 9')
ax10.set_xticks([0,5,10,15])
ax10.set_yticks([0,5,10,15])
plt.tight_layout(pad=2.0)
plt.show()

print('Steps 10-11: Let us now use these mean values in order to classify digits from the test data.')
print('The chosen measure is Euclidean distance.')

yhat = lib.euclidean_distance_classifier(X_test, means)
diff = yhat-y_test
success = (diff == 0).sum()/yhat.shape[0]
print('For example, the number 101 digit has been classified as {} and its actual value is {}.'.format(yhat[101],y_test[101]))
print('This is an example of a false classification. In general, the score of this classification process in terms of accuracy is {}%.'.format(success*100))

print('Step 12: Using these steps and the already defined functions, we may build a scikit-learn estimator-like class, called EuclideanDistanceClassifier.')
print('Let us demonstrate that the previous results remain the same, only now they are acquired in a more elegant fashion.')
clf = lib.EuclideanDistanceClassifier()
clf.fit(X_train,y_train)
score = clf.score(X_test,y_test)
print('The score of our classifier is {}%.'.format(score*100))
print('As expected, this is the same result as before.')

print('The final step of this lab preparation includes a 5-fold cross validation, as well as the depiction of the learning curve for this estimator.')
print('We shall use scikit-learn\'s built-in function for this purpose. We shall be attempting this process in two different ways:')
print('One using only the training data, and one by merging the train and test data together, in order to increase our data size.')

print('We begin with the results for the train data before merging.')

acc = lib.evaluate_euclidean_classifier(X_train, y_train)
print('The mean accuracy of this model is calculated to be {}%.'.format(acc*100))

train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
print('The learning curve for this model can be seen in this figure:')
lib.plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(.8, .9))

print('Now, we move on to the results for the merged data.')

new_X = np.concatenate((X_train, X_test), axis=0)
new_y = np.concatenate((y_train, y_test), axis=0)

acc_full = lib.evaluate_euclidean_classifier(new_X, new_y)
print('The mean accuracy of this model is calculated to be {}%.'.format(acc_full*100))

train_sizes, train_scores, test_scores = learning_curve(clf, new_X, new_y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
print('The learning curve for this model can be seen in this figure:')
lib.plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(.8, .9))

print('It appears that increasing the size of our data slightly reduces the model\'s accuracy and creates a somewhat bigger divergence between the training and the cv score, at least for small sample sizes.')

print('For the final part of the last step, we need to plot the decision surface for the classifier.')
print('To do that, we need to choose only 2 out of the 256 features (pixels). Doing it at random is ill-advised, so we shall be using scikit-learn\'s PCA algorithm.')

pca = PCA(n_components=2)
X_test_trans = pca.fit_transform(X_test)
X_train_trans = pca.fit_transform(X_train)

print('Of course, the model needs to be re-trained, this time using only the two features chosen by PCA.')
clf = lib.EuclideanDistanceClassifier()
clf.fit(X_train_trans,y_train)
print('The decision surface can be seen in this figure, along with the test data for each digit:')
lib.plot_clf(clf, X_test_trans, y_test)
print('The score of the re-trained Euclidean Classifier on the test data is {}%.'.format(100*clf.score(X_test_trans, y_test)))

print('Moving on, we need to construct a Naive Bayes Classifier, so for Step 14 we calculate the class priors.')
priors = lib.calculate_priors(X_train,y_train)

print('Using the function to calculate the class priors, we build a Naive Bayes Classifier from scratch.')
nbclf = lib.CustomNBClassifier()
nbclf.fit(X_train,y_train)
nbscore = nbclf.score(X_test,y_test)
print('The score of the classifier is {}%.'.format(nbscore*100))

train_sizes, train_scores, test_scores = learning_curve(nbclf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
print('The learning curve for this model can be seen in this figure:')
lib.plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0.35, 0.95))

print('However, notice that this score is dependent on the \'thermal\' parameter of our implementation.')
print('The calculation of the variance for each digit may result into some zeros and in the limit where σ = 0, the normal distrubution corresponds to a dirac delta function.')
print('In order to avoid division by zero, we introduce a parameter which is added to all of our variance calculations.')
print('This parameter follows the example of the scikit-learn GaussianNB function and is thus equal to a thermal constant (default value equal to 10^-9) times the highest value of the data variances.')
print('In this way, we do not significantly alter our calculated variances, but at the same time we manage to remove divisions by zero.')

print('An interesting idea would be to plot the score of this classifier based on different values of the thermal parameter.')
lib.thermal_investigation(X_train,y_train,X_test,y_test,50)

print('It becomes evident that our choice for this constant is detrimental to the model\'s accuracy, at least for our dataset.')
print('Let us compare our implementation to the one of scikit-learn\'s built-in function.')
autoclf = GaussianNB()
autoclf.fit(X_train, y_train)
print('The accuracy achieved is {}%.'.format(100*autoclf.score(X_test,y_test)))
print('Interestingly enough, it is exactly the same accuracy obtained for our model when setting the thermality constant equal to 10^-9.')
print('This is the reason why we have set it as the class\' default value.')

print('Another way of demonstrating the fact that our model\'s accuracy is highly dependent on the calculated variances is by setting the variances equal to 1 by default.')
nbclf = lib.CustomNBClassifier(use_unit_variance=True)
nbclf.fit(X_train,y_train)
print('The accuracy achieved this way is {}%.'.format(nbclf.score(X_test,y_test)*100))
print('One of the reasons why it\'s higher is because we did not have to find a workaround to avoid divisions by zero.')

print('Let us keep training models in order to evaluate them and compare their accuracy with our models\'.')
class_scores = lib.evaluate_all_classifiers(X_train,y_train)

for key in class_scores:
    print('The cross-validation score for the '+key+' classifier is {}%'.format(100*class_scores[key]))

print('Let us now move on to Step 18 - ensembles. First, we shall be creating a voting classifier, using some of the previously mentioned classifiers.')
print('In order to see which classifiers we will be combining for the voting system (odd number so that we can break ties), we must first plot the confusion matrix for each classifier mentioned, in order to ensure diversity in classification errors.')
lib.plot_confusion_matrices(X_train,y_train,X_test,y_test)

print('They are all good, so we choose Linear SVM, RBF SVM and kNN, only because there is no overlap between the (few) errors that they make.')

soft_estimators = [('rbf', SVC(probability=True, kernel="rbf")), ('linear', SVC(probability=True, kernel="linear")), ('knn', KNeighborsClassifier(n_neighbors=1))]
soft_scores = lib.evaluate_voting_classifier(soft_estimators,'soft',X_train,y_train)
print('The cross-validation score for the soft-voting ensemble is {}%.'.format(100*soft_scores))
hard_estimators = [('rbf', SVC(kernel="rbf")), ('linear', SVC(kernel="linear")), ('knn', KNeighborsClassifier(n_neighbors=1))]
hard_scores = lib.evaluate_voting_classifier(hard_estimators,'hard',X_train,y_train)
print('The cross-validation score for the hard-voting ensemble is {}%.'.format(100*hard_scores))

print('There indeed appears to be an improvement, but the margin for improvement was already small. Let\'s try this with estimators that do not score as well as these three.')

soft_estimators = [('nb', GaussianNB()), ('tree', DecisionTreeClassifier()), ('sigmoid', SVC(probability=True, kernel="sigmoid"))]
soft_scores = lib.evaluate_voting_classifier(soft_estimators,'soft',X_train,y_train)
print('The cross-validation score for the soft-voting ensemble is {}%.'.format(100*soft_scores))
hard_estimators = [('nb', GaussianNB()), ('tree', DecisionTreeClassifier()), ('sigmoid', SVC(kernel="sigmoid"))]
hard_scores = lib.evaluate_voting_classifier(hard_estimators,'hard',X_train,y_train)
print('The cross-validation score for the hard-voting ensemble is {}%.'.format(100*hard_scores))

print('While we\'re discussing ensembles, we may also implement a bagging classifier for each of the above estimators and see the result.')
scores = lib.evaluate_bagging_classifier(X_train,y_train)

for key in scores:
    print('The cross-validation score for the '+key+' Bagging classifier is {}%'.format(100*scores[key]))

clf = DecisionTreeClassifier()
accuracy = lib.evaluate_classifier(clf,X_train,y_train)
print('The accuracy of the decision tree is {}%.'.format(100*accuracy))

new_clf = BaggingClassifier(base_estimator=clf, n_estimators=10)
new_accuracy = lib.evaluate_classifier(new_clf,X_train,y_train)
print('The new accuracy of the decision tree is {}%.'.format(100*new_accuracy))

print('It\'s not coincidental that Bagging improves the Decision Tree model significantly.')
print('In fact, a more optimized case of Decision Tree with Bagging is what is known as Random Forest.')
score = lib.evaluate_random_forest_classifier(X_train,y_train)
print('The cross-validation score for the Random Forest model is {}%.'.format(100*score))

print('Let us now move on to the final part of this assignment (Step 19), the construction of a Neural Network.')

print('Our construction corresponds to a PyTorch model wrapped in a scikit-learn-style estimator.')
print('The inputs when calling the estimator are a list of layers, the number of features, the number of digits, the epochs, the batch size and the learning rate for the training.')
print('When fitting the model, there is an extra parameter called split. If split is set to zero, then the model is simply trained on the full training dataset.')
print('If split is set to a value between 0 and 1, then the training dataset is split into training and validation data, so a model.validation parameter is calculated, in order to give us an idea of our model\'s accuracy.')

print('For example, let us build a neural network with 2 sets of 100 hidden layers and train it for 300 epochs, with batch sizes equal to 32 and a learning rate of 0.01.')
nnclf = lib.PytorchNNModel([100,100], X_train.shape[1], 10, 300, 32, 1e-2)
nnclf.fit(X_train,y_train,split=0.2)
print('Setting the split parameter into 0.2 leads to a validation score of {}%.'.format(100*nnclf.validation))

print('This model has an accuracy of {}% for the actual test data.'.format(100*nnclf.score(X_test,y_test)))

print('Another way of validating our model is by the cross-validation function we have defined, which automatically splits the data into folds.')
nnscore = lib.evaluate_nn_classifier(X_train, y_train, [100,100], 300, 32, 1e-2, folds=5)

print('By cross-validation, the score of our neural network is {}%.'.format(100*nnscore))

print('Let us, finally, use our full training dataset to train our neural network and increase the number of epochs to 500.')
nnclf = lib.PytorchNNModel([100,100], X_train.shape[1], 10, 500, 32, 1e-2)
nnclf.fit(X_train,y_train,split=0.0)
print('The accuracy of this model on the actual test data is {}%.'.format(100*nnclf.score(X_test,y_test)))

train_sizes, train_scores, test_scores = learning_curve(nnclf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
print('The learning curve for this neural network is the following:')
lib.plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0.0, 1.0))