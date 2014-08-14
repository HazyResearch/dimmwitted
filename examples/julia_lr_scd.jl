
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
nfeat = 100
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
	@inbounds begin
		const label = row[length(row)]
		const nfeat = length(model)
		d = 0.0
		for i = 1:nfeat
			d = d + row[i]*model[i]
		end
	end
	return (-label * d + log(exp(d) + 1.0))
end

function grad(col::Array{Cdouble,1}, _colid::Cint, rows::Array{Array{Cdouble, 1}}, model::Array{Cdouble,1})
	colid = _colid + 1
	nfeat = length(model)
	nrows = length(rows)
	if colid > nfeat	# if it is the last column, then it is the labels, do nothing
		return 1.0
	end

	sum_term = 0.0
	pat_term = 0.0
	for ir = 1:length(rows)
		label = rows[ir][nfeat+1]
		d = 0.0
		for i = 1:nfeat
			d = d + rows[ir][i]*model[i]
		end
		sum_term = sum_term + label*rows[ir][colid]
		pat_term = pat_term + rows[ir][colid]*1.0/(1.0+exp(-d))
	end

	model[colid] = model[colid] - 0.00001* (-sum_term + pat_term)
	
	return 1.0
end

function avg(p_models::Ptr{Array{Cdouble,1}}, nrepl::Cint, irepl::Cint)
	models = pointer_to_array(p_models, convert(Int64, nrepl+1))
	julia_irepl = irepl + 1
	const nfeat = length(models[julia_irepl])
	for j = 1:nfeat
		d = 0.0
		for i = 1:nrepl
			d = d + models[i][j]
		end
		models[julia_irepl][j] = d/nrepl
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
                DimmWitted.AC_C2R)

######################################
# Register functions.
#
handle_loss = DimmWitted.register_row(dw, loss)
handle_grad = DimmWitted.register_c2r(dw, grad)
DimmWitted.register_model_avg(dw, handle_loss, avg, true)
DimmWitted.register_model_avg(dw, handle_grad, avg, true)

######################################
# Run 10 epoches.
#
for iepoch = 1:10
	rs = DimmWitted.exec(dw, handle_loss)
	println("LOSS: ", rs/nexp)
	println(sum(model))
	rs = DimmWitted.exec(dw, handle_grad)
end







