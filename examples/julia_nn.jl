push!(LOAD_PATH, "/home/shubham/Documents/research/dw/julialib/")
import DimmWitted
DimmWitted.set_libpath("/home/shubham/Documents/research/dw/libdw_julia")



######################################
# The following piece of code creates a
# synthetic data set:
#    - Data type is Cdouble
#    - Modle type is Array{Cdouble}
#
nexp = 1000
nfeat = 10
h = 5
examples = Array(Cdouble, nexp, nfeat+1)
for row = 1:nexp
	
	if rand() > 0.5
		examples[row, nfeat+1] = -1
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
model = Array[Cdouble[0 for j=1:nfeat+1] for i = 1:h]

function loss(row::Array{Cdouble,1}, model::Array{Array{Cdouble, 1},1})
	const nfeat = length(model) - 1
	d = 0.0
	for i = 1:nfeat
		d = d + row[i]*model[i]
	end
end

dw = DimmWitted.open(examples, model, 
                DimmWitted.MR_PERMACHINE,    
                DimmWitted.DR_SHARDING,      
                DimmWitted.AC_ROW)


