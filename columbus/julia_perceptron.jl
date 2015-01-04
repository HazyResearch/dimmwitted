#############################################################################################################
# MODULE : Implements Perceptron Classifier for Linear Classification of the given data
# 
# USAGE : Fitted_model, loss_value = perceptron(Data) 
# 
# INPUT : Data: M*(N+1) matrix 
#				It contains the data which is to be classified
#				For a given row, first N columns contain the features and last column 
#				contains actual labels
#
# OUTPUT : Fitted_model: Model fitted for classification of the data
#		   loss_value: Value of the perceptron loss function based on the final model
#

module julia_perceptron

export perceptron

import Base
path_lib = "$(dirname(Base.source_path()))/../libdw_julia"
path_dw = "$(dirname(Base.source_path()))/../julialib/"

push!(LOAD_PATH, path_dw)
import DimmWitted
DimmWitted.set_libpath(path_lib)


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
function perceptron(data)
	model = Cdouble[0 for i = 1:size(data,2)-1]
	nexp = size(data,1)
	dw = DimmWitted.open(data, model, 
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
	#######################################
	loss_value = 1
	for iepoch = 1:10
		rs = DimmWitted.exec(dw, handle_loss)
		println("LOSS: ", rs/nexp)
		loss_value = rs/nexp
		rs = DimmWitted.exec(dw, handle_update)
	end
	return model, loss_value
end

end