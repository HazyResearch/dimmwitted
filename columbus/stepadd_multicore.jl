#####################################################################################################################
# MODULE : Adds a feature to the given feature set based on the given data and a loss function using multiple cores
# 
# USAGE : A_new = add_feature(loss_func, A, fs, to_add)
# 
# INPUT : loss_func: Function to be used for measuring the performance of different candidate feature sets
#		  A: M*(N+1) matrix 
#				It contains the data from which we have to drop a feature out of the given feature set.
#				For a given row, first N columns contain all the features and last column contains 
#				actual labels/expected output
#		  fs: f*1 Array 
#				 Feature set to which a single feature is to be added. 'f' is the current number of features.
#		  to_add: 1 dimensional array 
#				 Feature set from which a feature is to be added to 'fs'. 
#	
# OUTPUT : A_new: A*(f+1) matrix, Data matrix with the new feature vector selected based on the loss function. 
#	   

module stepadd_multicore
export add_feature

using julia_ls
using DataFrames
using julia_ls
using julia_lr

#nexp = 100										
#nfeat = 10

#examples = Array(Cdouble, nexp, nfeat+1)
#for row = 1:nexp
#	for col = 1:nfeat
#		examples[row, col] = rand()
#	end
#	if rand() > 0.8
#		examples[row, nfeat+1] = 0
#	else
#		examples[row, nfeat+1] = 1
#	end
#end

#fs = [1,2,4,5]
#full = [1:nfeat]
#to_add = setdiff(full,fs)

function add_feature(func, A, fs,to_add)
	min_loss = Inf
	ncols = size(A,2)
	tasks = {A[:,union(fs,i,ncols)] for i in to_add}
	results = pmap(func, tasks)
	loss_value, idx = findmin([results[i][2] for i = 1:length(results)])
	model = results[idx][1]
	println("Feature Added", to_add[idx])
	return A[:,union(fs,to_add[idx])]
end

end