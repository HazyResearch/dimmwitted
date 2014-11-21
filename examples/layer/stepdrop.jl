push!(LOAD_PATH, "/home/shubham/Documents/research/dw/julialib/")
import DimmWitted
DimmWitted.set_libpath("/home/shubham/Documents/research/dw/libdw_julia")
import StatsBase
import Rmath

using julia_ls

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

###model = Cdouble[0 for i = 1:length]
###model = ls(examples, model)

fs = [1:nfeat]
A = examples
b = examples[:,end]

min_loss = Inf
tic()
loss_value = Array(Float32, length(fs))
for i = 1:length(fs)
	data = A[:,[1:i-1,i+1:end]]
	model = Cdouble[0 for i = 1:size(data,2)]
	model, loss_value = least_square(data, model)
	println(loss_value)
	if loss_value < min_loss
		min_loss = loss_value
		final = model
		global candidate = i
	end
end
toc()
println(candidate) 
println(min_loss)
