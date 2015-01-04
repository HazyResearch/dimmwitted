#############################################################################################################
# MODULE : Implements Logistic Regression based Model fitting algorithm for a given data
# 
# USAGE : Fitted_model, loss_value = logit_reg(Data) 
# 
# INPUT : Data: M*(N+1) matrix 
#				It contains the data on which the model is to be fitted
#				For a given row, first N columns contain the features and last column 
#				contains actual labels/expected output
#
# OUTPUT : Fitted_model: Model fitted on the data
#		   loss_value: Value of the log loss function based on the final model
#


module julia_lr

export logit_reg

import Base
path_lib = "$(dirname(Base.source_path()))/../libdw_julia"
path_dw = "$(dirname(Base.source_path()))/../julialib/"

push!(LOAD_PATH, path_dw)
import DimmWitted
DimmWitted.set_libpath(path_lib)


######################################
# The following piece of code creates a
# synthetic data set:
#    - Data type is Cdouble
#    - Modle type is Array{Cdouble}
#
#nexp = 100000
#nfeat = 1024
#examples = Array(Cdouble, nexp, nfeat+1)
#for row = 1:nexp
#	for col = 1:nfeat
#		examples[row, col] = 1
#	end
#	if rand() > 0.8
#		examples[row, nfeat+1] = 0
#	else
#		examples[row, nfeat+1] = 1
#	end
#end
#model = Cdouble[0 for i = 1:nfeat]

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

function grad(row::Array{Cdouble,1}, model::Array{Cdouble,1})
	const label = row[length(row)]
	const nfeat = length(model)
	d = 0.0
	for i = 1:nfeat
		d = d + row[i]*model[i]
	end
	d = exp(-d)
		Z = 0.001 * (-label + 1.0/(1.0+d))
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
function logit_reg(data)
	model = Cdouble[0 for i = 1:size(data,2)-1]	
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
	# Run 10 epoches.
	#
	loss_value = 1
	for iepoch = 1:10
		rs = DimmWitted.exec(dw, handle_loss)
		println("LOSS: ", rs/nexp)
		loss_value = rs/nexp
		rs = DimmWitted.exec(dw, handle_grad)
	end
	return model, loss_value
end

end