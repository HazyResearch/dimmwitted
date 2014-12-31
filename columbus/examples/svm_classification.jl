### Adding Folder containing Modules to the path ###
push!(LOAD_PATH, "/home/shubham/Documents/research/columbus/modules/")

### Importing Required Modules###
using julia_svm

### Initializing Data ###
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

#Last Column is the Expected Output
#Selecting a set of features for model estimation
fs = [1:NCols-1]
d = d[:,union(fs,NCols)]
println("Data Initialised")

### Model Fitting using SVM###
model_svm, loss_svm = svm(d)

### Output ###
println(loss_svm)

	