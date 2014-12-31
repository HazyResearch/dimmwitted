############################################################################################################
# MODULE : Drops a feature from the given feature set based on the given data and a loss function  
# 
# USAGE : A_new = drop_feature(loss_func, A, fs)
# 
# INPUT : loss_func: Function to be used for measuring the performance of different candidate feature sets
#		  A: M*(N+1) matrix 
#				It contains the data from which we have to drop a feature out of the given feature set.
#				For a given row, first N columns contain all the features and last column contains 
#				actual labels/expected output
#		  fs: f*1 Array 
#				 Feature set from which we have to drop a single feature
#	
# OUTPUT : A_new: A*(f-1) matrix, Data matrix after dropping a feature from the feature set 
#				  with the worst performance
#		   

module stepdrop

export drop_feature

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

###model = Cdouble[0 for i = 1:length]
###model = ls(examples, model)

#fs = [1:nfeat]
#A = examples
#b = examples[:,end]

function drop_feature(func,A,fs)
	ncols = size(A,2)
	A = A[:,union(fs,ncols)]
	min_loss = Inf
	loss_value = Array(Float32, length(fs))
	idx = 0
	for i = 1:length(fs)
		data = A[:,[1:i-1,i+1:end]]
		model, loss_value = func(data)
		println(loss_value)
		if loss_value < min_loss
			min_loss = loss_value
			final = model
			idx = i
		end
	end
	println(idx)
	return A[:,[1:idx-1,idx+1:size(fs,1)]]
end

end
