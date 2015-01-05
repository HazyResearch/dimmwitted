#####################################################################################################################
# MODULE : Implements Alternative Direction Method of Multipliers for Model fitting using Non linear loss functions
# 
# USAGE : 1. model = admm_qr(Data) [With QR factorization]
#         2. model = admm_noqr(Data) [Without QR factorization]
# 
# INPUT : Data: M*(N+1) matrix 
#				It contains the data on which the model is to be fitted
#				For a given row, first N columns contain the features and last column 
#				contains actual labels/expected output
#
# OUTPUT : model: Model fitted on the data using the non-linear loss function(Log loss function)
#

module julia_admm_c
export admm_qr, admm_noqr

function admm_qr(data)
	A = data[:,1:end-1]
	b = data[:,end]
	
	u = zeros(size(b))
	z = zeros(size(b))
	x = Cdouble[0 for i = 1:size(A,2)]

	LAMBDA = 0.001
	NEPOCH = 10

	@time Q,R = qr(A)
	for j = 1:NEPOCH
		println(j)
		x = \(R,transpose(Q)*(z-u))
		Ax = A*x
		pointer(z) = ccall((:cBISECT,"/home/shubham/Documents/research/columbus/c/cBisect"),Ptr{Float64},(Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Float64,),nrows,Ax,u,b,LAMBDA)
		u = u + Ax - z
	end
	return(x)
end

function admm_noqr(data)
	A = data[:,1:end-1]
	b = data[:,end]
		
	u = zeros(size(b))
	z = zeros(size(b))
	x = Cdouble[0 for i = 1:size(A,2)]
	nrows = size(b,1)
	println(typeof(nrows))
	LAMBDA = 0.001
	NEPOCH = 10

	AtA = (transpose(A)*A)
	for j = 1:NEPOCH
		println(j)
		x = \(AtA,transpose(A)*(z-u))
		Ax = A*x
		pointer(z) = ccall((:cBISECT,"/home/shubham/Documents/research/columbus/c/cBisect"),Ptr{Float64},(Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Float64,),nrows,Ax,u,b,LAMBDA)
		u = u + Ax - z
	end
	return(x)
end
end