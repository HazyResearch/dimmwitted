
push!(LOAD_PATH, "/Users/czhang/Desktop/Projects/dw_/julialib/")
import DimmWitted
DimmWitted.set_libpath("/Users/czhang/Desktop/Projects/dw_/libdw_julia")

immutable SHARED_DATA
	stepsize::Cdouble
	decay::Cdouble
end

shared_data = Array(SHARED_DATA, 1)
shared_data[1] = SHARED_DATA(0.00001, 0.99)

######################################
# The following piece of code creates a
# synthetic data set:
#    - Data type is Cdouble
#    - Modle type is Array{Cdouble}
#
nexp = 100000
nfeat = 10
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

######################################
# Define the loss function and gradient
# function for logistic regression
#
function loss(row::Array{Cdouble,1}, model::Array{Cdouble,1})
	const label = row[length(row)]
	const nfeat = length(model)
	d = 0.0
	for i = 1:nfeat
		d = d + row[i]*model[i]
	end
	return (-label * d + log(exp(d) + 1.0))
end

function grad(row::Array{Cdouble,1}, model::Array{Cdouble,1}, _shared_data::Array{SHARED_DATA,1})
	const stepsize = _shared_data[1].stepsize
	const label = row[length(row)]
	const nfeat = length(model)
	d = 0.0
	for i = 1:nfeat
		d = d + row[i]*model[i]
	end
	d = exp(-d)
	Z = stepsize * (-label + 1.0/(1.0+d))
  	for i = 1:nfeat
  		model[i] = model[i] - row[i] * Z
  	end
	return 1.0
end

######################################
# Create a DimmWitted object using data
# and model. You do not need to specify
# the type, they are infer'ed by the 
# open() function, which is parametric.
#
dw = DimmWitted.open(examples, model, 
                DimmWitted.MR_PERMACHINE,    
                DimmWitted.DR_SHARDING,      
                DimmWitted.AC_ROW, shared_data)

######################################
# Register functions.
#
handle_loss = DimmWitted.register_row(dw, loss)
handle_grad = DimmWitted.register_row2(dw, grad)
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







