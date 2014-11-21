module julia_ls

export least_square

push!(LOAD_PATH, "/home/shubham/Documents/research/dw/julialib/")
import DimmWitted
DimmWitted.set_libpath("/home/shubham/Documents/research/dw/libdw_julia")


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
function least_square(data, model)
	const nexp = size(data,1)
	dw = DimmWitted.open(data, model, 
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
	loss_value = 1
	for iepoch = 1:5
		rs = DimmWitted.exec(dw, handle_loss)
		println("LOSS: ", rs/nexp)
		loss_value = rs/nexp
		rs = DimmWitted.exec(dw, handle_grad)
	end
	return model, loss_value
end

end