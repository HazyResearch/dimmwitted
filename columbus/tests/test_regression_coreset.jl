push!(LOAD_PATH, "/home/shubham/Documents/research/columbus/modules/")
using julia_lr
using julia_ls
using julia_admm
using sampling

f = open("results_regression_coreset.txt","w")

NCols=151
NRows=100000
d = rand(NRows, NCols)
for row = 1:NRows
	if d[row, NCols] <= 0.5
		d[row, NCols] = 0
	else
		d[row, NCols] = 1
	end
end
fs = [1:NCols]
d = d[:,fs]

write(f, "Coreset Sampling: \n")
write(f, "Time Elapsed: ")
tic()
d,sampled = coreset(d)
write(f,string(toc(),"\n"))

write(f, "Size after Sampling: ")
write(f, string(size(d)))
write(f,"\n")

write(f, "Least Square with Coreset: \n")
write(f, "Time Elapsed: ")
tic()
@time least_square(d)
write(f,string(toc(),"\n"))

write(f,"\n")

write(f, "Logistic Regression with Coreset: \n")
write(f, "Time Elapsed: ")
tic()
@time logit_reg(d)
write(f,string(toc(),"\n"))

write(f, "Logistic Regression Using ADMM with Coreset and without QR: \n")
write(f, "Time Elapsed: ")
tic()
@time admm_noqr(d)
write(f,string(toc()),"\n")

write(f, "Logistic Regression Using ADMM with Coreset and QR: \n")
write(f, "Time Elapsed: ")
tic()
@time admm_qr(d)
write(f,string(toc()))
close(f)
