from random import seed
from random import randrange
import csv
from math import sqrt
import numpy as np

def load_csv(filename):
	dataset = list()
	count0,count1,count2 = 0,0,0
	with open(filename, 'r') as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			
			if count0+count1 == 3000:
 				break
			if not row:
				continue
			
			if row[-1] == '2' and count2 < 500:
				count2 += 1
			if row[-1] == '0' and count0 < 500:
				count0 += 1
				dataset.append(row)
			if row[-1] == '1' and count1 < 500:
				count1 += 1
				dataset.append(row)
	print (dataset[0])
	return dataset


def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	scores1=list()
	for fold in folds:
		fold=folds[0]
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted,newset,trees= algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	#predictions,newset=new_layer(trees,newset)
	#accuracy = accuracy_metric(actual, predictions)
	#scores1.append(accuracy)
	#print (scores1)
	return scores

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a dataset
def get_split(dataset, n_features):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	features = list()
	while len(features) < n_features:
		index = randrange(len(dataset[0])-1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	#print (group)
	f=0
	for j in group :
		f=f+1
		#print ("no of groups in leaf")
		#print (f)
	outcomes = [row[-1] for row in group]
	#print ("to_terminal")
	#print (outcomes)
	#print ("max")

	#print (max(set(outcomes), key=outcomes.count))
	return (outcomes)
	#return max(set(outcomes), key=outcomes.count)
	#return outcomes

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
	root = get_split(train, n_features)
	split(root, max_depth, min_size, n_features, 1)
	return root

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
	#predictions = [predict(tree, row) for tree in trees]
	#print (predictions)
	newset=list()
	predictions=list()
	prob1=list()
	predictions_prob=list()
	num=0
	for tree in trees:
		a=predict(tree,row)
		#converting to probabilities
		print ("outcomes of tree for particular row")
		print (a)
		proba=convert(a)
		print (proba)
		b=max(set(a), key=a.count)
		num=num+1
		#print (num)
		
		#print (proba)
		predictions_prob.append(a)
		predictions.append(b)
		prob1=prob1+proba
	print ("joint prob")
	print (prob1)

	return predictions_prob,max(set(predictions), key=predictions.count),prob1
	#return max(set(predictions), key=predictions.count)

# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
	trees = list()
	predictions=list()
	probablity=list()
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		tree = build_tree(sample, max_depth, min_size, n_features)
		trees.append(tree)
	f=0
	for j in test :
		f=f+1
		#print (f)
	#predictions = layer(test,trees)
	newset=list()
	predictions,newset=new_layer(trees,test)
	#for row in test:
	#	prob,pred,train = bagging_predict(trees, row)
	#	newset.append(train)
		
	#	predictions.append(pred)
	#	probablity.append(prob)
	#print (newset)
	return predictions,newset,trees

def new_layer(trees,test):
	newset=list()
	predictions=list()
	probability=list()
	for row in test:
		prob,pred,train = bagging_predict(trees, row)
		newset.append(train)
		
		predictions.append(pred)
		probability.append(prob)
	print (newset)
	return predictions,newset

def convert(prob):
	classes=[0,1]

	size=0
	for row in prob:
		size=size+1
	#print ("size")
	#print (size)
	#print (prob)

	probability=list()
	for class_val in classes:
		p=0
		for row in prob:
			
			if int(row)==int(class_val):
				p=p+1
		q=p/size
		probability.append(q)

	#print (probability)
	return probability

#def layer(test,trees):
#	predictions=list()
#	probablity=list()
#	for row in test:
#		prob,pred = bagging_predict(trees, row)
#		predictions.append(pred)
#		probablity.append(prob)
#	print (predictions)



seed(2)
filename = 'train.csv'
dataset = load_csv(filename)






#a, b = 0 , 19
#fold[b], fold[a] = fold[a], fold[b]

#for row in dataset:
#	row[b], row[a] = row[a], row[b]
#	print (row[b])
		



n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_trees=5
n_features = int(sqrt(len(dataset[0])-1))
scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
print('Trees: %d' % n_trees)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
