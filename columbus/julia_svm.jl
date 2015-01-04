#############################################################################################################
# MODULE : Implements SVM Classifier for Linear Classification of the given data
# 
# USAGE : Fitted_model, loss_value = svm(Data) 
# 
# INPUT : Data: M*(N+1) Array
#				It contains the data which is to be classified
#				For a given row, first N columns contain the features and last column 
#				contains actual labels
#
# OUTPUT : Fitted_model: Model fitted for classification of the data
#		   loss_value: Value of the hinge loss function based on the final model
#

module julia_svm
export svm

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
#    - Model type is Array{Cdouble}
#

#nexp = 100000
#nfeat = 1024
#examples = Array(Cdouble, nexp, nfeat+1)
#for row = 1:nexp
#	
#	if rand() > 0.5
#		examples[row, nfeat+1] = -1
#		for col = 1:nfeat
#			examples[row, col] = -1.0*col/nfeat
#		end
#	else
#		examples[row, nfeat+1] = 1
#		for col = 1:nfeat
#			examples[row, col] = col*1.0/nfeat
#		end
#	end
#end
#model = Cdouble[0 for i = 1:nfeat]

C = 10.0
lambda = 2/C
######################################
# Define the loss(hinge) function and gradient
# function for SVM
#
function loss(row::Array{Cdouble,1}, model::Array{Cdouble,1})
	const label = row[length(row)]
	const nfeat = length(model)
	d = 0.0
	for i = 1:nfeat
		d = d + row[i]*model[i]
	end
	loss1 = 0.0
	for i = 1:nfeat
		loss1 = loss1 + model[i]*model[i]
	end
	loss2 = 0 > (1-label*d) ? 0.0: (1-label*d)*1.0 
	return (0.2*loss1*0.5 + loss2)
end

function grad(row::Array{Cdouble,1}, model::Array{Cdouble,1})
	const label = row[length(row)]
	const nfeat = length(model)
	d = 0.0
	for i = 1:nfeat
		d = d + row[i]*model[i]
	end

  	for i = 1:nfeat
		if label*d < 1  		
			model[i] = model[i] - 0.001*(0.2*model[i] - label*row[i])
		else 
			model[i] = model[i] - 0.001*0.2*model[i]
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
function svm(data)
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
	handle_grad = DimmWitted.register_row(dw, grad)

	println("dimmwitted registered")
	#######################################
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