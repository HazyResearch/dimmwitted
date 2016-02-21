
module DimmWitted

immutable DW
  _dw::Ptr{Void}
  modelrepl::Cint
  datarepl::Cint
  accessmethod::Cint
  shareddatatype
end

_libpath = ""
_data_type = ""
_model_type = ""

_nogc = Any[0.0,0.0]

MR_PERCORE = 0
MR_PERNODE = 1
MR_PERMACHINE = 2
MR_SINGLETHREAD_DEBUG = 3

DR_FULL = 0
DR_SHARDING = 1

AC_ROW = 0
AC_COL = 1
AC_C2R = 2

function set_libpath(path)
	global _libpath
	_libpath = path
end   

function get_libpath()
	global _libpath
	return _libpath
end

function hi()
	@eval ccall(($(string("Hello")), $(_libpath)), Void, ()) 
	return nothing
end


function open{DATATYPE, MODELTYPE}(examples::SparseMatrixCSC{DATATYPE,Int64}, model::Array{MODELTYPE,1}, _modelrepl, _datarepl, _acmethod, _shared_data=[1,2])

	modelrepl = convert(Cint, _modelrepl)
	datarepl = convert(Cint, _datarepl)
	acmethod = convert(Cint, _acmethod)

	global _libpath, _dw, _data_type, _model_type, _nogc

	_model = model

	nrows = examples.m
	ncols = examples.n
	colptr = examples.colptr
	rowptr = examples.rowval
	nnz = examples.nzval

	@eval immutable tmptype
		idx::Clonglong
		data::$(DATATYPE)
	end

	append!(_nogc, Any[_model, examples, _shared_data, tmptype])

	nmodelel = size(model, 1)

	_data_type = tmptype
	_model_type = MODELTYPE
	_shared_tupe = typeof(_shared_data)

	t = _shared_tupe
	n = length(_shared_data)

	_dw = @eval ccall( ($(string("SparseDimmWitted_Open2")), $(_libpath)), Ptr{Void}, (Any, Any, Any, Clonglong, Clonglong, Clonglong, Clonglong, Ptr{Void}, Ptr{Clonglong}, Ptr{Clonglong}, Ptr{Void}, Cint, Cint, Cint, Any, Cint, Ptr{Void}), $(Array{DATATYPE}), $(Array{tmptype}), $(Array{MODELTYPE}), $(nrows), $(ncols), $(length(nnz)), $(nmodelel), $(nnz), $(rowptr), $(colptr) ,$(_model), $(modelrepl), $(datarepl), $(acmethod), $(t), $(n), $(_shared_data))

	dw = DW(_dw, modelrepl, datarepl, acmethod, typeof(_shared_data))

	println("[JULIA-DW] Created Sparse DimmWitted Object: ", dw._dw)

	return dw
end



function open{DATATYPE, MODELTYPE}(examples::Array{DATATYPE,2}, model::Array{MODELTYPE,1}, _modelrepl, _datarepl, _acmethod, _shared_data=[1,2])

	modelrepl = convert(Cint, _modelrepl)
	datarepl = convert(Cint, _datarepl)
	acmethod = convert(Cint, _acmethod)

	global _libpath, _dw, _data_type, _model_type, _nogc

	_examples = examples
	_model = model
	_examples_c = examples.'

	append!(_nogc, Any[_examples, _model, _examples_c, _shared_data])
	
	nrows = size(examples, 1)
	ncols = size(examples, 2)
	nmodelel = size(model, 1)
	
	_data_type = DATATYPE
	_model_type = MODELTYPE
	_shared_tupe = typeof(_shared_data)

	t = _shared_tupe
	n = length(_shared_data)
	
	_dw = @eval ccall( ($(string("DenseDimmWitted_Open2")), $(_libpath)), Ptr{Void}, (Any, Any, Clonglong, Clonglong, Clonglong, Ptr{Void}, Ptr{Void}, Cint, Cint, Cint, Any, Cint, Ptr{Void}), $(Array{DATATYPE}), $(Array{MODELTYPE}), $(nrows), $(ncols), $(nmodelel), $(_examples_c), $(_model), $(modelrepl), $(datarepl), $(acmethod), $(t), $(n), $(_shared_data))
	
	dw = DW(_dw, modelrepl, datarepl, acmethod, typeof(_shared_data))

	println("[JULIA-DW] Created Dense DimmWitted Object: ", dw._dw)

	return dw
end

function check_is_safe(func, ret, parameter)

	const stdout = STDOUT
	const rd, wr = redirect_stdout()
	code_llvm(func, parameter)
	str = readavailable(rd)
	close(rd)
	redirect_stdout(STDERR)

	#if contains(str, "alloc") || contains(
	#		replace(
	#			replace(str, string("julia_",func), ""),
	#			"julia_type", "")
	#	, "julia_")
	#	println(str)
	#	return false
	#else
	#	return true
	#end
	return true
end


function set_n_numa_nodes(_dw, n_numa_nodes)
	@eval ccall(($(string("set_n_numa_node")), $(_libpath)), Cuint, (Ptr{Void}, Cint, Cint, Cint, Cint), $(_dw._dw), $(n_numa_nodes), $(_dw.modelrepl), $(_dw.datarepl), $(_dw.accessmethod)) 
end

function set_n_threads_per_node(_dw, n_thread_per_node)
	@eval ccall(($(string("set_n_thread_per_node")), $(_libpath)), Cuint, (Ptr{Void}, Cint, Cint, Cint, Cint), $(_dw._dw), $(n_thread_per_node), $(_dw.modelrepl), $(_dw.datarepl), $(_dw.accessmethod)) 
end

function register_row2(_dw, func, supress=false)

	global _data_type, _model_type, _libpath, _nogc

	is_safe = check_is_safe(func, Cdouble, (Array{_data_type,1}, Array{_model_type,1}, _dw.shareddatatype))
	if is_safe == false && supress==false
		error("Your function contains LLVM LR `alloc` or `call` other julia functions. We cannot register this function because it protentially is not thread-safe. Use register_row(_dw",",",func,",true) to register this function AT YOUR OWN RISK!")
	end

	const func_c = cfunction(func, Cdouble, (Array{_data_type,1}, Array{_model_type,1}, _dw.shareddatatype))

	append!(_nogc, Any[func_c, func])

	handle = @eval ccall(($(string("DenseDimmWitted_Register_Row2")), $(_libpath)), Cuint, (Ptr{Void}, Ptr{Void}, Cint, Cint, Cint), $(_dw._dw), $(func_c), $(_dw.modelrepl), $(_dw.datarepl), $(_dw.accessmethod)) 

	println("[JULIA-DW] Registered Row Function ", func, " Handle=", handle)

	return handle
end


function register_row(_dw, func, supress=false)

	global _data_type, _model_type, _libpath, _nogc

	is_safe = check_is_safe(func, Cdouble, (Array{_data_type,1}, Array{_model_type,1}))
	if is_safe == false && supress==false
		error("Your function contains LLVM LR `alloc` or `call` other julia functions. We cannot register this function because it protentially is not thread-safe. Use register_row(_dw",",",func,",true) to register this function AT YOUR OWN RISK!")
	end


	const func_c = cfunction(func, Cdouble, (Array{_data_type,1}, Array{_model_type,1}))

	append!(_nogc, Any[func_c, func])

	handle = @eval ccall(($(string("DenseDimmWitted_Register_Row2")), $(_libpath)), Cuint, (Ptr{Void}, Ptr{Void}, Cint, Cint, Cint), $(_dw._dw), $(func_c), $(_dw.modelrepl), $(_dw.datarepl), $(_dw.accessmethod)) 

	println("[JULIA-DW] Registered Row Function ", func, " Handle=", handle)

	return handle
end

function register_c2r(_dw, func, supress=false)

	is_safe = check_is_safe(func, Cdouble, (Array{_data_type,1}, Cint, Array{Array{Cdouble, 1},1}, Array{_model_type,1}))
	if is_safe == false && supress==false
		error("Your function contains LLVM LR `alloc` or `call` other julia functions. We cannot register this function because it protentially is not thread-safe. Use register_c2r(_dw",",",func,",true) to register this function AT YOUR OWN RISK!")
	end

	global _data_type, _model_type, _libpath, _nogc

	const func_c = cfunction(func, Cdouble, (Array{_data_type,1}, Cint, Array{Array{Cdouble, 1},1}, Array{_model_type,1}))

	append!(_nogc, Any[func_c, func])

	handle = @eval ccall(($(string("DenseDimmWitted_Register_C2R2")), $(_libpath)), Cuint, (Ptr{Void}, Ptr{Void}, Cint, Cint, Cint), $(_dw._dw), $(func_c), $(_dw.modelrepl), $(_dw.datarepl), $(_dw.accessmethod)) 

	println("[JULIA-DW] Registered Column-to-row Function ", func, " Handle=", handle)

	return handle
end

function register_c2r2(_dw, func, supress=false)

	is_safe = check_is_safe(func, Cdouble, (Array{_data_type,1}, Cint, Array{Array{Cdouble, 1},1}, Array{_model_type,1}, _dw.shareddatatype))
	if is_safe == false && supress==false
		error("Your function contains LLVM LR `alloc` or `call` other julia functions. We cannot register this function because it protentially is not thread-safe. Use register_c2r(_dw",",",func,",true) to register this function AT YOUR OWN RISK!")
	end

	global _data_type, _model_type, _libpath, _nogc

	const func_c = cfunction(func, Cdouble, (Array{_data_type,1}, Cint, Array{Array{Cdouble, 1},1}, Array{_model_type,1}, _dw.shareddatatype))

	append!(_nogc, Any[func_c, func])

	handle = @eval ccall(($(string("DenseDimmWitted_Register_C2R2")), $(_libpath)), Cuint, (Ptr{Void}, Ptr{Void}, Cint, Cint, Cint), $(_dw._dw), $(func_c), $(_dw.modelrepl), $(_dw.datarepl), $(_dw.accessmethod)) 

	println("[JULIA-DW] Registered Column-to-row Function ", func, " Handle=", handle)

	return handle
end


function register_col(_dw, func, supress=false)

	is_safe = check_is_safe(func, Cdouble, (Array{_data_type,1}, Array{_model_type,1}, _dw.shareddatatype))
	if is_safe == false && supress==false
		error("Your function contains LLVM LR `alloc` or `call` other julia functions. We cannot register this function because it protentially is not thread-safe. Use register_row(_dw",",",func,",true) to register this function AT YOUR OWN RISK!")
	end

	global _data_type, _model_type, _libpath, _nogc

	const func_c = cfunction(func, Cdouble, (Array{_data_type,1}, Array{_model_type,1}, _dw.shareddatatype))

	append!(_nogc, Any[func_c, func])

	handle = @eval ccall(($(string("DenseDimmWitted_Register_Col2")), $(_libpath)), Cuint, (Ptr{Void}, Ptr{Void}, Cint, Cint, Cint), $(_dw._dw), $(func_c), $(_dw.modelrepl), $(_dw.datarepl), $(_dw.accessmethod)) 

	println("[JULIA-DW] Registered Col Function ", func, " Handle=", handle)

	return handle
end

function register_col(_dw, func, supress=false)

	is_safe = check_is_safe(func, Cdouble, (Array{_data_type,1}, Array{_model_type,1}))
	if is_safe == false && supress==false
		error("Your function contains LLVM LR `alloc` or `call` other julia functions. We cannot register this function because it protentially is not thread-safe. Use register_row(_dw",",",func,",true) to register this function AT YOUR OWN RISK!")
	end

	global _data_type, _model_type, _libpath, _nogc

	const func_c = cfunction(func, Cdouble, (Array{_data_type,1}, Array{_model_type,1}))

	append!(_nogc, Any[func_c, func])

	handle = @eval ccall(($(string("DenseDimmWitted_Register_Col2")), $(_libpath)), Cuint, (Ptr{Void}, Ptr{Void}, Cint, Cint, Cint), $(_dw._dw), $(func_c), $(_dw.modelrepl), $(_dw.datarepl), $(_dw.accessmethod)) 

	println("[JULIA-DW] Registered Col Function ", func, " Handle=", handle)

	return handle
end


function register_model_avg(_dw, handle, func, supress=false)

	is_safe = check_is_safe(func, Cdouble, (Array{Array{_model_type,1},1}, Cint, Cint))
	if is_safe == false && supress==false
		error("Your function contains LLVM LR `alloc` or `call` other julia functions. We cannot register this function because it protentially is not thread-safe. Use register_model_avg(_dw",",",func,",true) to register this function AT YOUR OWN RISK!")
	end

	global _data_type, _model_type, _libpath, _nogc

	const func_c = cfunction(func, Cdouble, (Array{Array{Cdouble,1},1}, Cint, Cint))

	append!(_nogc, Any[func_c, func])

	@eval ccall(($(string("DenseDimmWitted_Register_ModelAvg2")), $(_libpath)), Void, (Ptr{Void}, Cuint, Ptr{Void}, Cint, Cint, Cint), $(_dw._dw), $(handle), $(func_c), $(_dw.modelrepl), $(_dw.datarepl), $(_dw.accessmethod))

	println("[JULIA-DW] Registered Avg Function ", func, " for Func Handle=", handle)

	return nothing

end

function exec(_dw, func_handle)
	global _libpath

	rs = @eval ccall(($(string("DenseDimmWitted_Exec2")), $(_libpath)), Cdouble, (Ptr{Void}, Cuint, Cint, Cint, Cint), $(_dw._dw), $(func_handle), $(_dw.modelrepl), $(_dw.datarepl), $(_dw.accessmethod))

	return rs
end


export libpath, get_libpath, open, register_row, exec

end


