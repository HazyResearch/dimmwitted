
push!(LOAD_PATH, "/Users/czhang/Desktop/Projects/dw_/julialib/")
import DimmWitted
DimmWitted.set_libpath("/Users/czhang/Desktop/Projects/dw_/libdw_julia")

######################################
# The following piece of code creates a
# synthetic data set:
#    - Data type is Cdouble
#    - Modle type is Array{Cdouble}
#
nexp = 100000
nfeat = 1024
examples = Array(Cdouble, nexp, nfeat+1)
for row = 1:nexp
	for col = 1:nfeat
		examples[row, col] = 1
	end
	if rand() > 0.8
		examples[row, nfeat+1] = 0
	else
		examples[row, nfeat+1] = 1
	end
end
model = Cdouble[0 for i = 1:nfeat]

sparse_example=sparse(examples)

######################################
# Define the loss function and gradient
# function for logistic regression
#
function loss(row, model::Array{Cdouble,1})
	const nfeat = length(model)
	const lastcol = nfeat + 1
	const nnz = length(row)
	label = 0.0
	if row[nnz].idx == lastcol
		label = row[nnz].data
	end
	d = 0.0
	for i = 1:nnz
		if row[i].idx != lastcol
			d = d + row[i].data*model[row[i].idx]
		end
	end
	return (-label * d + log(exp(d) + 1.0))
end

function grad(row, model::Array{Cdouble,1})
	const nfeat = length(model)
	const lastcol = nfeat + 1
	const nnz = length(row)
	label = 0.0
	if row[nnz].idx == lastcol
		label = row[nnz].data
	end
	d = 0.0
	for i = 1:nnz
		if row[i].idx != lastcol
			d = d + row[i].data*model[row[i].idx]
		end
	end
	d = exp(-d)
	Z = 0.00001 * (-label + 1.0/(1.0+d))
	for i = 1:nnz
		if row[i].idx != lastcol
		model[row[i].idx] = model[row[i].idx] - row[i].data * Z
		end
	end
	return 1.0
end

######################################
# Create a DimmWitted object using data
# and model. You do not need to specify
# the type, they are infer'ed by the 
# open() function, which is parametric.
#
dw = DimmWitted.open(sparse_example, model, 
                DimmWitted.MR_PERMACHINE,    
                DimmWitted.DR_SHARDING,      
                DimmWitted.AC_ROW)

######################################
# Register functions.
#
handle_loss = DimmWitted.register_row(dw, loss)
handle_grad = DimmWitted.register_row(dw, grad)
#DimmWitted.register_model_avg(dw, handle_loss, avg, true)
#DimmWitted.register_model_avg(dw, handle_grad, avg, true)

######################################
# Run 10 epoches.
#
for iepoch = 1:10
	rs = DimmWitted.exec(dw, handle_loss)
	println("LOSS: ", rs/nexp)
	rs = DimmWitted.exec(dw, handle_grad)
end







