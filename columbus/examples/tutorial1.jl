### Adding Folder containing Modules to the path ###
push!(LOAD_PATH, "/home/shubham/Documents/research/columbus/modules/")

### Importing Required Modules###
using DataFrames
using julia_ls

input_file = "/home/shubham/Documents/research/data/household_power_consumption.txt"
data = readtable(input_file ,separator=';',decimal='.')
println(data[1,:])

data = zeros(Any,5,1)
open(input_file,"r") do f
	row = 1
    for line in eachline(f)
    	if row == 1
    		row = row+1
    		continue
    	end
      	feature = float(split(line,";")[[3,7:9]])
      	label = feature[1]*1000/60 - feature[2] - feature[3] - feature[4]
      	data = hcat(data,[feature, label])
      	println(transpose(data))
    end
end

println(data)
	