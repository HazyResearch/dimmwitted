push!(LOAD_PATH, "/home/shubham/Documents/research/dw/julialib/")
import DimmWitted
DimmWitted.set_libpath("/home/shubham/Documents/research/dw/libdw_julia")

######################################
# The following piece of code creates a
# synthetic data set:
#    - Data type is Cdouble
#    - Model type is Array{Cdouble}
#

nexp = 100000
nfeat = 1024
learning_rate = 0.6
examples = Array(Cdouble, nexp, nfeat+2)

for row = 1:nexp
	examples[row,nfeat+1] = 1
	if rand() > 0.5
		examples[row, nfeat+2] = 0
		for col = 1:nfeat
			examples[row, col] = -1.0*col/nfeat	
		end
	else
		examples[row, nfeat+1] = 1
		for col = 1:nfeat
			examples[row, col] = col*1.0/nfeat
			
		end
	end
end
model = Cdouble[0 for i = 1:(nfeat+1)]

######################################
# Define the loss function and weight update
# function for Perceptron
#
function loss(row::Array{Cdouble,1}, model::Array{Cdouble,1})
	const label = row[length(row)]
	const nfeat = length(model)
	d = 0.0
	for i = 1:nfeat
		d = d + row[i]*model[i]
	end
	pred = d >= 0 ? 1:0
	return 0.6*(label - pred)
end

function update(row::Array{Cdouble,1}, model::Array{Cdouble,1})
	const label = row[length(row)]
	const nfeat = length(model)
	d = 0.0
	for i = 1:nfeat
		d = d + row[i]*model[i]
	end
	pred = d >= 0 ? 1:0
  	for i = 1:nfeat
		model[i] = model[i] + 0.6*(label - pred)*row[i]
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
handle_update = DimmWitted.register_row(dw, update)

println("dimmwitted registered")
######################################
																																																																																																																																																																																																																																																																																		# Run 10 epoches.
#
for iepoch = 1:10
	rs = DimmWitted.exec(dw, handle_loss)
	println("LOSS: ", rs/nexp)
	rs = DimmWitted.exec(dw, handle_update)
end
