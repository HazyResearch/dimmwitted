#addprocs(7)
@everywhere using stepdrop_multicore
@everywhere using stepadd_multicore
using stepdrop
using transform
using sampling
using julia_svm
using julia_perceptron
using julia_admm

println("initialized")

nexp = 100
nfeat = 10

examples = Array(Cdouble, nexp, nfeat+1)
for row = 1:nexp
	for col = 1:nfeat
		examples[row, col] = rand()
	end
	if rand() > 0.8
		examples[row, nfeat+1] = 0
	else
		examples[row, nfeat+1] = 1
	end
end

fs = [1:nfeat]
A = examples
b = examples[:,end]
#fs = [1,2,4,5]
full = [1:nfeat]
to_add = setdiff(full,fs)

println("dataset created")

#A_drop = drop_feature(least_square,A,fs)
x_admm = admm(A[:,fs],b)
#A_drop2 = drop_multicore(A,fs)
#A_add = add_feature(A,fs,to_add)
#A_sampled = coreset(A,fs)
