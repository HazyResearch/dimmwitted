##### Refer http://julialang.org/blog/2013/04/distributed-numerical-optimization/ ######

@everywhere using julia_ls

#for i in 1:7
#	addprocs(i)
#end

@everywhere using julia_ls
@everywhere using DataFrames
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

fs = [1,2,4,5]
full = [1:nfeat]
to_add = setdiff(full,fs)

min_loss = Inf
println("sampled")
tasks = {A[:,union(fs,i)] for i in to_add}
model = {Cdouble[0 for j = 1:(length(fs)+1)] for i = 1:length(to_add)}
println("danger initialised")
tic()
results = pmap(least_square, tasks, model)
loss_value, candidate = findmin([results[i][2] for i = 1:length(results)])
toc()
model = results[candidate][1]
println("danger tackled")
println(to_add[candidate])

