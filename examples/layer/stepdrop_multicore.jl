##### Refer http://julialang.org/blog/2013/04/distributed-numerical-optimization/ ######

@everywhere using julia_ls

#for i in 1:7
#	addprocs(i)
#end

@everywhere using julia_ls

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

println("initialised")

#model = Cdouble[0 for i = 1:length]
#model = ls(examples, model)

A = examples
#b = examples[:,end]
fs = A[1,1:end-1]

min_loss = Inf
println("sampled")
tasks = {A[:,[1:i-1,i+1:size(A,2)]] for i = 1:length(fs)}
model = {Cdouble[0 for j = 1:(length(fs)-1)] for i = 1:length(fs)}
println("danger initialised")
tic()
results = pmap(least_square, tasks, model)
loss_value, candidate = findmin([results[i][2] for i = 1:length(results)])
toc()
model = results[candidate][1]
println("danger tackled")
println(candidate)

