module transform

export union, select, remove

function union(fs1 , fs2)
	return union(fs1, fs2)
end

function select(A, fs1)
	return  A[:,fs1]
end

function remove(fs1,fs2)
	return setdiff(fs1,fs2)
end

