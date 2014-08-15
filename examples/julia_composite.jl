
push!(LOAD_PATH, "/Users/czhang/Desktop/Projects/dw_/julialib/")
import DimmWitted
DimmWitted.set_libpath("/Users/czhang/Desktop/Projects/dw_/libdw_julia")

immutable DoublePair
	d1::Cdouble
	d2::Cdouble
end

######################################
# The following piece of code creates a
# synthetic data set:
#    - Data type is Cdouble
#    - Modle type is Array{Cdouble}
#
nexp = 100000
nfeat = 1024
examples = Array(DoublePair, nexp, nfeat+1)
for row = 1:nexp
	for col = 1:nfeat
		examples[row, col] = DoublePair(1.0,1.0)
	end
	if rand() > 0.8
		examples[row, nfeat+1] = DoublePair(1.0,0.0)
	else
		examples[row, nfeat+1] = DoublePair(0.0,1.0)
	end
end
model = DoublePair[DoublePair(0.0,0.0) for i = 1:nfeat]

######################################
# Define the loss function and gradient
# function for logistic regression
#
function loss(row::Array{DoublePair,1}, model::Array{DoublePair,1})
	const label1 = row[length(row)].d1
	const label2 = row[length(row)].d2
	const nfeat = length(model)
	d1 = 0.0
	d2 = 0.0
	for i = 1:nfeat
		d1 = d1 + row[i].d1*model[i].d1
		d2 = d2 + row[i].d2*model[i].d2
	end
	v1 = (-label1 * d1 + log(exp(d1) + 1.0))
	v2 = (-label2 * d2 + log(exp(d2) + 1.0))
	return v1 + v2
end

function grad(row::Array{DoublePair,1}, model::Array{DoublePair,1})
	const label1 = row[length(row)].d1
	const label2 = row[length(row)].d2
	const nfeat = length(model)
	d1 = 0.0
	d2 = 0.0
	for i = 1:nfeat
		d1 = d1 + row[i].d1*model[i].d1
		d2 = d2 + row[i].d2*model[i].d2
	end
	d1 = exp(-d1)
	d2 = exp(-d2)
	Z1 = 0.00001 * (-label1 + 1.0/(1.0+d1))
	Z2 = 0.00001 * (-label2 + 1.0/(1.0+d2))
  	for i = 1:nfeat
  		model[i] = DoublePair(model[i].d1 - row[i].d1 * Z1, model[i].d2 - row[i].d2 * Z2)
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
                DimmWitted.AC_ROW)

######################################
# Register functions.
#
handle_loss = DimmWitted.register_row(dw, loss, true)
handle_grad = DimmWitted.register_row(dw, grad, true)
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

sum1 = 0.0
sum2 = 0.0
for i = 1:length(model)
	sum1 = sum1 + model[i].d1
	sum2 = sum2 + model[i].d2
end
println("SUM OF MODEL1: ", sum1)
println("SUM OF MODEL2: ", sum2)





