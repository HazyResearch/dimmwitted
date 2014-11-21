push!(LOAD_PATH, "/home/shubham/Documents/research/dw/julialib/")
import DimmWitted
using julia_ls
DimmWitted.set_libpath("/home/shubham/Documents/research/dw/libdw_julia")

nexp = 100000
nfeat = 1024
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
#model = Cdouble[0 for i = 1:nfeat]

A = examples[:,1:nfeat]
b = examples[:,end]

u = zeros(size(b))
z = zeros(size(b))
x = zCdouble[0 for i = 1:nfeat]

lambda = 0.001
nepoch = 10

for i = 1:epoch
	println(i)
	data = hcat(A,(z-u))
	x = least_square(data,x)
	Ax = A*x
	z = bisect()