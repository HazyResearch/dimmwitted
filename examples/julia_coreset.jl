#push!(LOAD_PATH, "/home/shubham/Documents/research/dw/julialib/")
#import DimmWitted
#DimmWitted.set_libpath("/home/shubham/Documents/research/dw/libdw_julia")
import StatsBase
import Rmath

######################################
# The following piece of code creates a
# synthetic data set:
#    - Data type is Cdouble
#    - Model type is Array{Cdouble}
#

nexp = 10
nfeat = 2
examples = Array(Cdouble, nexp, nfeat+1)
for row = 1:nexp
	for col = 1:nfeat
		examples[row, col] = rand(1:9)/10
	end
	if rand() > 0.8
		examples[row, nfeat+1] = 0
	else
		examples[row, nfeat+1] = 1
	end
end
model = Cdouble[0 for i = 1:nfeat]
##println(model)

function coreset(data)
	inv_data = inv(*(transpose(data),data))
	sensitivity1 = sum((*(data,inv_data) .* data), 1)
	sensitivity = Float64[x for x in sensitivity1]
	weight = StatsBase.WeightVec(sensitivity)
	println(sum(sensitivity))
	sampled = StatsBase.sample(1:length(sensitivity), weight, int(2*(sum(sensitivity)-1)*100))
	data_sampled = data[sampled,:]
	return data_sampled
end

examples_sampled = coreset(examples)

#println(model)
println(length(examples_sampled))
