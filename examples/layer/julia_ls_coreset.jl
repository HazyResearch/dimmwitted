push!(LOAD_PATH, "/home/shubham/Documents/research/dw/julialib/")
import DimmWitted
DimmWitted.set_libpath("/home/shubham/Documents/research/dw/libdw_julia")
import StatsBase
import Rmath

######################################
# The following piece of code creates a
# synthetic data set:
#    - Data type is Cdouble
#    - Model type is Array{Cdouble}
#

nexp = 100000															
nfeat = 1000
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

model = Cdouble[0 for i = 1:nfeat]

println("Data Initialised")

function coreset(data)
	features = data[:,1:(size(data,2)-1)]
	label = data[:,size(data,2)]
	m = *(transpose(features),features)
	inv_features = inv(m)
	sensitivity1 = sum((*(features,inv_features) .* features), 1)
	sensitivity = Float64[x for x in sensitivity1]
	#println(sensitivity)
	weight = StatsBase.WeightVec(sensitivity)
	sampled = StatsBase.sample(1:length(sensitivity), weight, int(2*(sum(sensitivity)-1)*100))
	println(length(sampled))
	if length(sampled) < size(data,1)
		data_sampled = data[sampled,:]
	else
		data_sampled = data	
	end
	return data_sampled
end

#examples = coreset(examples)
println("Data Sampled Using Coreset")
######################################
# Define the loss function and gradient
# function for linear regression
#
function loss(row::Array{Cdouble,1}, model::Array{Cdouble,1})
	const label = row[length(row)]
	const nfeat = length(model)
	d = 0.0
	for i = 1:nfeat
		d = d + row[i]*model[i]
	end
	return (0.5 * ((d - label)^2))
end

function grad(row::Array{Cdouble,1}, model::Array{Cdouble,1})
	const label = row[length(row)]
	const nfeat = length(model)
	d = 0.0
	for i = 1:nfeat
		d = d + row[i]*model[i]
	end
	Z = 0.001 * (d - label)
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
                DimmWitted.MR_SINGLETHREAD_DEBUG,    
                DimmWitted.DR_SHARDING,      
                DimmWitted.AC_ROW)

println("dimmWitted opened")
######################################
# Register functions.
#
handle_loss = DimmWitted.register_row(dw, loss)
handle_grad = DimmWitted.register_row(dw, grad)

println("dimmwitted registered")
######################################
#####Run 10 epoches.

for iepoch = 1:5
	rs = DimmWitted.exec(dw, handle_loss)
	println("LOSS: ", rs/nexp)
	rs = DimmWitted.exec(dw, handle_grad)
end

