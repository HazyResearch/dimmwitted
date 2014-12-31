### Adding Folder containing Modules to the path ###
push!(LOAD_PATH, "/home/shubham/Documents/research/columbus/modules/")

### Importing Required Modules###
using julia_ls

############################# Synthetic Data #################################
### Initializing Data: Expected Output = Average of features ###
#Last Column is the Expected Output

NFeat=10
NRows=100000
data = rand(NRows, NFeat+1)
for row = 1:NRows
  data[row, NFeat+1] = mean(data[row,1:end-1])
end
println("Data Initialised")

### Model Fitting using Least Square Regression###
model_ls, loss_ls = least_square(data)

### Output ###
println("Model:")
println(model_ls)

############################# Real Data ####################################
### Reading Data from a CSV file ####
input_file = "/home/shubham/Documents/research/data/gasoline.csv"
data = readcsv(input_file)

##### Breaking the data into features and expected output and normalizing the data #####
feature = data[:, 2:4]
feature = (feature .- minimum(feature,1))./(maximum(feature,1) - minimum(feature,1))
output = data[:, 6]
output = (output .- minimum(output,1))./(maximum(output,1) - minimum(output,1))
data = hcat(feature,ones(size(feature,1)),output)

### Model Fitting using Least Square Regression###
model_ls, loss_ls = least_square(data)

### Output ###
println(model_ls)