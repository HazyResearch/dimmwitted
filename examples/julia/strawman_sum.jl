import Base.@time

nnumbers = 100000000
numbers = Float64[ i-1 for i = 1:nnumbers ]
tasks = Int64[ i-1 for i = 1:nnumbers ]


output = Float64[0.0]
@time ccall( (:percore_sum, "/Users/czhang/Desktop/dw/libdw_julia"), Void, (Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Int32), numbers, output, tasks, nnumbers) 
println("Result=", output[1])

output = Float64[0.0]
@time ccall( (:strawman_sum, "/Users/czhang/Desktop/dw/libdw_julia"), Void, (Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Int32), numbers, output, tasks, nnumbers) 
println("Result=", output[1])

output = Float64[0.0]
@time ccall( (:hogwild_sum, "/Users/czhang/Desktop/dw/libdw_julia"), Void, (Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Int32), numbers, output, tasks, nnumbers) 
println("Result=", output[1])

output = Float64[0.0]
@time ccall( (:percore_sum, "/Users/czhang/Desktop/dw/libdw_julia"), Void, (Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Int32), numbers, output, tasks, nnumbers) 
println("Result=", output[1])



println("Correct Result=", sum(numbers))

#function mycompare(example, model, label, nfeat)
#	dot = 0.0
#	for i = 1:nfeat
#		dot = dot + example[i] * model[i]
#	end
#	d = exp(dot)
#	Z = -label + d/(1.0+d)
#	for i = 1:nfeat
#		model[i] = model[i] - 0.0001*example[i]*Z
#	end
#end
