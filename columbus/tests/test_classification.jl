push!(LOAD_PATH, "/home/shubham/Documents/research/columbus/modules/")
using julia_perceptron
using julia_svm

f = open("results_classification.txt","w")

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

write(f, "Perceptron Algorithm Without Materialization: \n")
write(f, "Time Elapsed: ")
tic()
@time perceptron(d)
write(f,string(toc(),"\n"))

write(f,"\n")

write(f, "SVM Without Materialization: \n")
write(f, "Time Elapsed: ")
tic()
@time svm(d)
write(f,string(toc()))

close(f)
