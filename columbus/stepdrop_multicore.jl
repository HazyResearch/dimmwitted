#######################################################################################################################
# MODULE : Drops a feature from the given feature set based on the given data and a loss function using multiple cores 
# 
# USAGE : A_new = drop_feature_multicore(loss_func, A, fs)
# 
# INPUT : loss_func: Function to be used for measuring the performance of different candidate feature sets
#		  A: M*(N+1) matrix 
#				It contains the data from which we have to drop a feature out of the given feature set.
#				For a given row, first N columns contain all the features and last column contains 
#				actual labels/expected output
#		  fs: f*1 Array 
#				 Feature set from which we have to drop a single feature
#	
# OUTPUT : A_new: A*(f-1) matrix, Data matrix after dropping a feature, with the worst performance, from the feature set
#

module stepdrop_multicore
export drop_feature_multicore

using julia_ls

#nexp = 100										
#nfeat = 10
#
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

#A = examples
#fs = A[1,1:end-1]

function drop_feature_multicore(func, A, fs)
	min_loss = Inf
	ncols = size(A,2)
	A = A[:,union(fs,ncols)]
	tasks = {A[:,[1:i-1,i+1:size(A,2)]] for i = 1:length(fs)}
	
	results = pmap(func, tasks)
	loss_value, idx = findmin([results[i][2] for i = 1:length(results)])
	model = results[idx][1]

	A_drop = A[:,[1:idx-1,idx+1:size(fs,1)]]
	println("Feature Dropped: ", fs[idx])
	return A_drop
end
end
