addprocs(7)
push!(LOAD_PATH, "/home/shubham/Documents/research/columbus/modules/")
using julia_ls
using julia_lr
using julia_admm
using stepdrop
@everywhere using stepdrop_multicore
@everywhere using stepadd_multicore

f = open("results_features.txt","w")

NCols=151
NRows=1000
d = rand(NRows, NCols)
fs = [1,3,4,5,6,7,9,12,13,14,50,60]

write(f, "StepDrop using Least Squares Regression: \n")
write(f, "Time Elapsed: ")
tic()
@time drop_feature_multicore(least_square, d, fs)
write(f,string(toc(),"\n"))

write(f,"\n")

write(f, "StepDrop using Logistic Regression: \n")
write(f, "Time Elapsed: ")
tic()
@time drop_feature_multicore(logit_reg, d, fs)
write(f,string(toc()))

write(f,"\n")

fs2 = [100,101,102,103,104,105,106,107,108]

write(f, "StepAdd using Least Squares Regression: \n")
write(f, "Time Elapsed: ")
tic()
@time add_feature(least_square, d, fs, fs2)
write(f,string(toc(),"\n"))

write(f,"\n")

write(f, "StepAdd using Logistic Regression: \n")
write(f, "Time Elapsed: ")
tic()
@time add_feature(logit_reg, d, fs, fs2)
write(f,string(toc()))

write(f,"\n")
close(f)
