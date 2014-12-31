###############################################################################################
# MODULE: Module for various manual feature selection/modification
# FUNCTIONS - 
# 1. JOIN - Function for joining two feature sets 
# USAGE : fs_new = join(fs1, fs2)
# INPUT : fs1 and fs2 are the two feature vectors to be joined. These can be single dimensional Arrays/vectors/sets.
# OUTPUT : fs_new : Feature vector which is the union of the two input feature vectors
#
# 2. REMOVE - Function for removing a set of features from another set
# USAGE: fs_new = remove(fs1,fs2)
# INPUT: fs1 - Set of features from which a fraction of features are to be removed
#		 fs2 - Set of features to be removed from fs2
# OUTPUT: fs_new : Set of features which is the difference of the two input feature sets	
#
# 3. SELECT - Projection of a Data Matrix based on the given feature set
# USAGE: Ap = select(A, fs1)
# INPUT: A - M*N Array 
#		 fs1 - Set of features of dimension f*1, where f is the number of features to be selected
# OUTPUT: Ap - Projected matrix of dimension M*f
#

module transform

export union, select, remove

function join(fs1 , fs2)
	return union(fs1, fs2)
end

function select(A, fs1)
	return  A[:,fs1]
end

function remove(fs1,fs2)
	return setdiff(fs1,fs2)
end

end

