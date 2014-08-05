
nexp = 10000
nfeat = 10

examples = Float64[ 1 for i = 1:nexp*nfeat ]
labels = Float64[1 for i = 1:nexp]
model = Float64[0 for i = 1:nfeat]




function mycompare(example, model, label, nfeat)
	println("~~~~~~")
end


const mycompare_c = cfunction(mycompare, Void, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint))


@time ccall( (:glm_sgd, "/Users/czhang/Desktop/dw/libdw_julia"), Void, (Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Int64, Int64, Ptr{Void}), examples,
labels, model, nexp, nfeat, mycompare_c) 
