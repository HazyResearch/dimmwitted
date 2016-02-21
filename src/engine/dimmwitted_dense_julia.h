// Copyright 2014 Hazy Research (http://i.stanford.edu/hazy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef _DENSE_DW_JULIA_H
#define _DENSE_DW_JULIA_H

#include "util.h"
#include "dimmwitted_const.h"
#include "dimmwitted_dstruct.h"

#include "engine/scheduler.h"
#include "engine/scheduler_strawman.h"
#include "engine/scheduler_hogwild.h"
#include "engine/scheduler_percore.h"
#include "engine/scheduler_pernode.h"

#include "julia.h"
//#include "julia_internal.h"


struct SparsePair{
	long long idx;
	double data;
};


/**
 * \brief Task description for the row-access method.
 *
 * \tparam A type of data elements (See DenseDimmWitted)
 * \tparam B type of model (See DenseDimmWitted)
 */
template<class B>
class TASK_ROW_JULIA{
public:
	jl_array_t* row_pointers; /**<A list of pointer to rows*/
	jl_array_t * shared_data;
	double (*f) (const jl_array_t* const, B* const, jl_array_t *);	/**<Row-access function to execute*/
	TASK_ROW_JULIA(jl_array_t* const _row_pointers, 
			double (* _f) (const jl_array_t* const p_row, B* const p_model, jl_array_t *),
			jl_array_t * _shared_data
			):
		row_pointers(_row_pointers), f(_f), shared_data(_shared_data)
	{}
	TASK_ROW_JULIA(){}
};

/**
 * \brief Task description for the column-access method.
 *
 * \tparam A type of data elements (See DenseDimmWitted)
 * \tparam B type of model (See DenseDimmWitted)
 */
template<class B>
class TASK_COL_JULIA{
public:
	jl_array_t * shared_data;
	jl_array_t* col_pointers; /**<A list of pointer to columns*/
	double (*f) (const jl_array_t* const, B* const, jl_array_t *);	/**<Column-access function to execute*/
	TASK_COL_JULIA(jl_array_t* const _col_pointers,
		double (*_f) (const jl_array_t* const, B* const, jl_array_t *),
		jl_array_t * _shared_data):
		col_pointers(_col_pointers), f(_f), shared_data(_shared_data)
	{}
	TASK_COL_JULIA(){}
};

/**
 * \brief Task description for the column-to-row-access method.
 *
 * \tparam A type of data elements (See DenseDimmWitted)
 * \tparam B type of model (See DenseDimmWitted)
 */
template<class B>
class TASK_C2R_JULIA{
public:
	jl_array_t * shared_data;
	jl_array_t*  col_pointers; /**<A list of pointer to columns*/
	Pair<long, jl_array_t*> *  c2r_col_2_rowbuffer_idxs; /**<See DenseDimmWitted::c2r_col_2_rowbuffer_idxs*/
	jl_array_t**  c2r_row_pointers_buffer; /**<See DenseDimmWitted::_c2r_row_pointers_buffer*/
	double (*f) (const jl_array_t* const p_col, int i_col,
									const jl_array_t* const p_rows, B* const, jl_array_t *);
									/**<Column-to-row-access function to execute*/

	TASK_C2R_JULIA(jl_array_t* _col_pointers, Pair<long, jl_array_t*> * _c2r_col_2_rowbuffer_idxs,
		jl_array_t** _c2r_row_pointers_buffer, 
		double (*_f) (const jl_array_t* const p_col, int i_col,
					const jl_array_t* const p_rows, B* const, jl_array_t *),
		jl_array_t * _shared_data):
		col_pointers(_col_pointers), c2r_col_2_rowbuffer_idxs(_c2r_col_2_rowbuffer_idxs),
		c2r_row_pointers_buffer(_c2r_row_pointers_buffer), f(_f), shared_data(_shared_data)
	{}

	TASK_C2R_JULIA(){}

};

/**
 * \brief A thin wrapper for the row-access function that can be used by DWRun (engine/scheduler.h).
 *
 * \tparam A type of data elements (See DenseDimmWitted)
 * \tparam B type of model (See DenseDimmWitted)
 */
template<class B>
double dense_map_row_JULIA (long i_task, const TASK_ROW_JULIA<B> * const rddata, B * const wrdata){
	return rddata->f(&rddata->row_pointers[i_task], wrdata, rddata->shared_data);
}

/**
 * \brief A thin wrapper for the column-access function that can be used by DWRun (engine/scheduler.h).
 *
 * \tparam A type of data elements (See DenseDimmWitted)
 * \tparam B type of model (See DenseDimmWitted)
 */
template<class B>
double dense_map_col_JULIA (long i_task, const TASK_COL_JULIA<B> * const rddata, B * const wrdata){
	return rddata->f(&rddata->col_pointers[i_task], wrdata, rddata->shared_data);
}

/**
 * \brief A thin wrapper for the column-to-row-access function that can be used by DWRun (engine/scheduler.h).
 *
 * \tparam A type of data elements (See DenseDimmWitted)
 * \tparam B type of model (See DenseDimmWitted)
 */
template<class B>
double dense_map_c2r_JULIA (long i_task, const TASK_C2R_JULIA<B> * const rddata, B * const wrdata){
	const Pair<long, jl_array_t*> & row_ptrs = rddata->c2r_col_2_rowbuffer_idxs[i_task];
	//std::cout << "###" << row_ptrs.first << "     " << row_ptrs.second << std::endl;
	//std::cout << "!!!" << (void*) rddata->c2r_col_2_rowbuffer_idxs << std::endl;

	return rddata->f(&rddata->col_pointers[i_task], i_task, row_ptrs.second, wrdata, rddata->shared_data);
	return 1.0;
}

/**
 * \brief A thin wrapper for allocating model replicas that can be used by DWRun (engine/scheduler.h).
 *
 * \tparam A type of data elements (See DenseDimmWitted)
 * \tparam B type of model (See DenseDimmWitted)
 */
template<class B>
void dense_model_allocator_JULIA (jl_array_t ** const a, const jl_array_t * const b){
	(*a) = (jl_array_t*) ::operator new(((sizeof(jl_array_t)+jl_array_ndimwords(1)*sizeof(size_t)+15)&-16));
	//(*a)->type = (b)->type;
	(*a)->data = (void *) new char[(b)->length*(b)->elsize];
	(*a)->length = (b)->length;
	(*a)->elsize = (b)->elsize;
	(*a)->ptrarray = (b)->ptrarray;
	(*a)->ndims = (b)->ndims;
	(*a)->isshared = (b)->isshared;
	(*a)->isaligned = (b)->isaligned;
	(*a)->how = (b)->how;
	(*a)->nrows = (b)->nrows;
	(*a)->maxsize = (b)->maxsize;
	(*a)->offset = (b)->offset;
	memcpy((*a)->data, (b)->data, (b)->length*(b)->elsize);
}

/**
 * \brief This class is the interface of DimmWitted for Dense data.
 * For sparse data, you can use engine/dimmwitted_sparse.h
 *
 * \tparam A Type of elements for data.
 * \tparam B Type of model.
 * \tparam model_repl_type Model replication strategy.
 * \tparam data_repl_type Data replication strategy.
 * \tparam access_mode Access Method
 */
template<class B, ModelReplType model_repl_type, DataReplType data_repl_type, AccessMode access_mode>
class DenseDimmWitted_Julia{
public:

	ModelReplType _model_repl_type;

	DataReplType _data_repl_type;

	AccessMode _access_mode;

	typedef double (*DW_FUNCTION_ROW) (const jl_array_t* const, B* const, jl_array_t *);
		/**<\brief Type of row-access function*/

	typedef double (*DW_FUNCTION_COL) (const jl_array_t* const, B* const, jl_array_t *);
		/**<\brief Type of column-access function*/

	typedef double (*DW_FUNCTION_C2R) (const jl_array_t* const p_col, int i_col,
														const jl_array_t* const p_rows, B* const, jl_array_t *);
		/**<\brief Type of column-to-row-access function*/

	typedef void (*DW_FUNCTION_MAVG) (jl_array_t* const p_models, int nreplicas, int ireplica);
		/**<\brief Type of the function that conducts model averaging*/

	char* const p_data; /**<\brief Pointer to the data, which is a two dimensional array of type A*/

	long n_rows; /**<\brief Number of rows of the data.*/

	long n_cols; /**<\brief Number of columns of the data.*/

	long n_nzelems;

	long * p_rows;

	long * p_cols;
	
	B* const p_model; /**<\brief Pointer to the model.*/

	std::map<unsigned int, DW_FUNCTION_ROW> fs_row; /**<\brief Map from function handle to row-access functions.*/

	std::map<unsigned int, DW_FUNCTION_COL> fs_col; /**<\brief Map from function handle to col-access functions.*/

	std::map<unsigned int, DW_FUNCTION_C2R> fs_c2r; /**<\brief Map from function handle to column-to-row-access functions.*/

	std::map<unsigned int, DW_FUNCTION_MAVG> fs_avg; /**<\brief Map from function handle to model averaging functions.
                                                       Note that, a model averaging function does not have
                                                       its own handle, the function handle in this map is
                                                       the handle for row-access/col-access/column-to-row-access
                                                       functions. DimmWitted will use the corresponding averaging
                                                       function when these function handles are used.
                                                    */
	
	unsigned int current_handle_id;	/**<\brief Function handle for the next function-registering.*/

	jl_array_t* const row_pointers; /**<\brief Pointer to each row of the data.*/

	jl_array_t* const col_pointers; /**<\brief Pointer to each column of the data.*/

	jl_array_t** _c2r_row_pointers_buffer; /**<\brief For column-to-row-access, this data structure
                                              stores the following information. For a matrix
                                              \verbatim
                                                     Col1 Col2 Col3
                                                Row1 0    A12  A13
                                                Row2 A21  0    A23
                                              \endverbatim
                                              This contains a list of row pointers
                                              \verbatim
                                                Row2 Row1 Row1 Row2
                                              \endverbatim
                                              Where the first Row2 is for Col1, and Row1 is not
                                              here because A11=0. The second Row1 is for
                                              Col2, and the last Row1 Row2 is for Col3.
                                            */

	Pair<long, jl_array_t*> * const c2r_col_2_rowbuffer_idxs; /**<\brief For For column-to-row-access, this data structure
                                              stores the following information. Still use the
                                              example of DenseDimmWitted::_c2r_row_pointers_buffer, for this example
                                              matrix, this data structure contains
                                              \verbatim
                                                (0,1), (1,1), (2,2)
                                              \endverbatim
                                              Where
                                              \verbatim
                                                (0,1) corresponds to Row2
                                                (1,1) corresponds to Row1
                                                (2,2) corresponds to Row1, Row2
                                              \endverbatim
                                              More generally, (a,b) defines a sequence
                                              in DenseDimmWitted::_c2r_row_pointers_buffer, starting from
                                              the position a, and contain b elements.
                                              */

    char * _sparse_buffer_row;

	char * _new_ele_buffer; /**<\brief If the input is row-wise storage, and the user requires column-wise
                         access, this pointer points to the re-structured region of
                         the memory for column-wise storage*/

	long * row_ids; /**<\brief A list of indexes that the system will access. For row-access, it will be
                  [0,nrows]. */

	long * col_ids; /**<\brief A list of indexes that the system will access. For column-access or 
									 column-to-row access, it will be [0,ncols]. */

	TASK_ROW_JULIA<B> task_row; /**<\brief Task description for row-access. */

	TASK_COL_JULIA<B> task_col; /**<\brief Task description for column-access. */

	TASK_C2R_JULIA<B> task_c2r; /**<\brief Task description for column-to-row access. */

	DWRun<TASK_ROW_JULIA<B>, B, model_repl_type, data_repl_type> dw_row_runner;
		/**<\brief Execution backend for row-access. See engine/scheduler.h. */

	DWRun<TASK_COL_JULIA<B>, B, model_repl_type, data_repl_type> dw_col_runner;
		/**<\brief Execution backend for column-access. See engine/scheduler.h. */

	DWRun<TASK_C2R_JULIA<B>, B, model_repl_type, data_repl_type> dw_c2r_runner;
		/**<\brief Execution backend for column-to-row-access. See engine/scheduler.h. */

	jl_array_t * shared_data;

public:

	/**
	 * \brief Constructor of dense DimmWitted.
	 *
	 * \param _data The data as a row-wise stored two dimensional array of type A.
	 * \param _n_rows Number of rows in the data.
	 * \param _n_cols Number of columns in the data.
	 * \param _model Pointer to the model.
	 */
	DenseDimmWitted_Julia(void * _data, jl_value_t *data_el_type, 
						  long _n_rows, long _n_cols, B * const _model, jl_array_t * _shared_data):
		p_data((char*)_data), p_model(_model),
		n_rows(_n_rows), n_cols(_n_cols), 
		current_handle_id(0),
		row_pointers((jl_array_t*) ::operator new(_n_rows * ((sizeof(jl_array_t)+jl_array_ndimwords(1)*sizeof(size_t)+15)&-16))),
		col_pointers((jl_array_t*) ::operator new(_n_cols * ((sizeof(jl_array_t)+jl_array_ndimwords(1)*sizeof(size_t)+15)&-16))),
		c2r_col_2_rowbuffer_idxs(new Pair<long, jl_array_t*>[_n_cols]),
		row_ids(new long[_n_rows]),
		col_ids(new long[_n_cols]),
		dw_row_runner(DWRun<TASK_ROW_JULIA<B>, B, model_repl_type, data_repl_type>(&task_row, p_model, dense_model_allocator_JULIA<B>)),
		dw_col_runner(DWRun<TASK_COL_JULIA<B>, B, model_repl_type, data_repl_type>(&task_col, p_model, dense_model_allocator_JULIA<B>)),
		dw_c2r_runner(DWRun<TASK_C2R_JULIA<B>, B, model_repl_type, data_repl_type>(&task_c2r, p_model, dense_model_allocator_JULIA<B>)),
		_model_repl_type(model_repl_type),
		_data_repl_type(data_repl_type),
		_access_mode(access_mode),
		shared_data(_shared_data)
	{

		dw_row_runner.isjulia = true;
		dw_col_runner.isjulia = true;
		dw_c2r_runner.isjulia = true;

		for(int i=0;i<n_rows;i++){
			row_ids[i] = i;
		}
		for(int j=0;j<n_cols;j++){
			col_ids[j] = j;
		}

		size_t elsz = jl_datatype_size(data_el_type);
		if(access_mode == DW_ACCESS_ROW || access_mode == DW_ACCESS_C2R){
			for(int i=0;i<n_rows;i++){
					//row_pointers[i].type = data_el_type;
					row_pointers[i].data = (void*) &(p_data[i*elsz*n_cols]);
					row_pointers[i].length = n_cols;
					row_pointers[i].elsize = elsz;
					row_pointers[i].ptrarray = false;
					row_pointers[i].ndims = 1;
					row_pointers[i].isshared = 1;
					row_pointers[i].isaligned = 0;
					row_pointers[i].how = 0;
					row_pointers[i].nrows = n_cols;
					row_pointers[i].maxsize = n_cols;
					row_pointers[i].offset = 0;
					//row_pointers[i] = DenseVector<A>(p_data[i], n_cols);
			}
			task_row.row_pointers = row_pointers;
			task_row.shared_data = shared_data;
			dw_row_runner.prepare();
		}
		if(access_mode == DW_ACCESS_COL || access_mode == DW_ACCESS_C2R){
			_new_ele_buffer = new char[n_rows*n_cols*elsz];
			long ct = 0;
			for(int j=0;j<n_cols;j++){
				//col_pointers[j].type = data_el_type;
				col_pointers[j].data = (void*) &_new_ele_buffer[ct*elsz];
				col_pointers[j].length = n_rows;
				col_pointers[j].elsize = elsz;
				col_pointers[j].ptrarray = false;
				col_pointers[j].ndims = 1;
				col_pointers[j].isshared = 1;
				col_pointers[j].isaligned = 0;
				col_pointers[j].how = 0;
				col_pointers[j].nrows = n_rows;
				col_pointers[j].maxsize = n_rows;
				col_pointers[j].offset = 0;
				for(int i=0;i<n_rows;i++){
					//_new_ele_buffer[ct++] = _data[i][j];
					memcpy((void*)&_new_ele_buffer[(ct++)*elsz], &(p_data[(i*n_cols+j)*elsz]), elsz);
				}
			}
			task_col.col_pointers = col_pointers;
			task_col.shared_data = shared_data;
			dw_col_runner.prepare();

			if(access_mode == DW_ACCESS_C2R){

				char * testblock = new char[elsz];
				memset (testblock, 0, elsz);

				_c2r_row_pointers_buffer = new jl_array_t*[n_rows*n_cols]; // ::operator new(n_rows * n_cols * ((sizeof(jl_array_t)+jl_array_ndimwords(1)*sizeof(size_t)+15)&-16));
				long ct = 0;
				for(int j=0;j<n_cols;j++){
					c2r_col_2_rowbuffer_idxs[j].first = ct;
					int a = 0;
					for(int i=0;i<n_rows;i++){

						if (memcmp (testblock, &((char*)col_pointers[j].data)[i*elsz], elsz) != 0){
							a++;

							_c2r_row_pointers_buffer[ct] = &row_pointers[i];

							ct ++;
						}
					}

					c2r_col_2_rowbuffer_idxs[j].second = (jl_array_t*) ::operator new(((sizeof(jl_array_t)+jl_array_ndimwords(1)*sizeof(size_t)+15)&-16));
					//c2r_col_2_rowbuffer_idxs[j].second->type = jl_typeof(&row_pointers[0]);
					c2r_col_2_rowbuffer_idxs[j].second->data = (void*) &_c2r_row_pointers_buffer[c2r_col_2_rowbuffer_idxs[j].first];
					c2r_col_2_rowbuffer_idxs[j].second->length = a;
					c2r_col_2_rowbuffer_idxs[j].second->elsize = sizeof(void*);
					c2r_col_2_rowbuffer_idxs[j].second->ptrarray = true;
					c2r_col_2_rowbuffer_idxs[j].second->ndims = 1;
					c2r_col_2_rowbuffer_idxs[j].second->isshared = 1;
					c2r_col_2_rowbuffer_idxs[j].second->isaligned = 0;
					c2r_col_2_rowbuffer_idxs[j].second->how = 0;
					c2r_col_2_rowbuffer_idxs[j].second->nrows = a;
					c2r_col_2_rowbuffer_idxs[j].second->maxsize = a;
					c2r_col_2_rowbuffer_idxs[j].second->offset = 0;

				}

				task_row.row_pointers = row_pointers;
				task_row.shared_data = shared_data;
				dw_row_runner.prepare();

				task_c2r.col_pointers = col_pointers;
				task_c2r.c2r_col_2_rowbuffer_idxs = c2r_col_2_rowbuffer_idxs;
				task_row.shared_data = shared_data;

				task_c2r.c2r_row_pointers_buffer = _c2r_row_pointers_buffer;
				dw_c2r_runner.prepare();

			}

		}
	}

	void set_n_numa_node(int n_numa_node){
		dw_row_runner.n_numa_node = n_numa_node;
		dw_col_runner.n_numa_node = n_numa_node;
		dw_c2r_runner.n_numa_node = n_numa_node;
	}

	void set_n_thread_per_node(int n_thread_per_node){
		dw_row_runner.n_thread_per_node = n_thread_per_node;
		dw_col_runner.n_thread_per_node = n_thread_per_node;
		dw_c2r_runner.n_thread_per_node = n_thread_per_node;
	}

	/**
	 * Note that this is CSC from Julia
	 */
	DenseDimmWitted_Julia(void * _data, long * _rows, long * _cols, jl_value_t *data_el_type, 
							jl_value_t *wrapped_el_type,
						  long _n_rows, long _n_cols, long _n_nzelems, B * const _model, jl_array_t * _shared_data):
		p_data((char*)_data), p_model(_model),
		n_rows(_n_rows), n_cols(_n_cols), n_nzelems(_n_nzelems),
		p_rows(_rows), p_cols(_cols),
		current_handle_id(0),
		row_pointers((jl_array_t*) ::operator new(_n_rows * ((sizeof(jl_array_t)+jl_array_ndimwords(1)*sizeof(size_t)+15)&-16))),
		col_pointers((jl_array_t*) ::operator new(_n_cols * ((sizeof(jl_array_t)+jl_array_ndimwords(1)*sizeof(size_t)+15)&-16))),
		c2r_col_2_rowbuffer_idxs(new Pair<long, jl_array_t*>[_n_cols]),
		row_ids(new long[_n_rows]),
		col_ids(new long[_n_cols]),
		dw_row_runner(DWRun<TASK_ROW_JULIA<B>, B, model_repl_type, data_repl_type>(&task_row, p_model, dense_model_allocator_JULIA<B>)),
		dw_col_runner(DWRun<TASK_COL_JULIA<B>, B, model_repl_type, data_repl_type>(&task_col, p_model, dense_model_allocator_JULIA<B>)),
		dw_c2r_runner(DWRun<TASK_C2R_JULIA<B>, B, model_repl_type, data_repl_type>(&task_c2r, p_model, dense_model_allocator_JULIA<B>)),
		_model_repl_type(model_repl_type),
		_data_repl_type(data_repl_type),
		_access_mode(access_mode),
		shared_data(_shared_data)
	{

		dw_row_runner.isjulia = true;
		dw_col_runner.isjulia = true;
		dw_c2r_runner.isjulia = true;

		for(int i=0;i<n_rows;i++){
			row_ids[i] = i;
		}
		for(int j=0;j<n_cols;j++){
			col_ids[j] = j;
		}

		size_t elsz = jl_datatype_size(data_el_type);
		size_t wrapper_elsz = jl_datatype_size(wrapped_el_type);
		if(access_mode == DW_ACCESS_ROW || access_mode == DW_ACCESS_C2R){

			long * rowcounts = new long[n_rows];
			long * rowpointers = new long[n_rows];
			long * rowpointers_ori = new long[n_rows];
			for(int i=0;i<n_rows;i++){
				rowcounts[i] = 0;
			}
			int n = 0;
			for(int i=0;i<n_cols;i++){
				if(i==n_cols-1){
					n = n_nzelems - (p_cols[i]-1);
				}else{
					n = p_cols[i+1] - p_cols[i];
				}
				for(int j=0;j<n;j++){
					rowcounts[p_rows[p_cols[i]-1+j]-1] ++;
				}
			}

			for(int i=0;i<n_rows;i++){
				rowpointers[i] = 0;
			}

			for(int i=1;i<n_rows;i++){
				rowpointers[i] = rowpointers[i-1] + rowcounts[i-1];
				rowpointers_ori[i] = rowpointers[i];
			}
			rowpointers_ori[0] = rowpointers[0];
			_sparse_buffer_row = new char[n_nzelems*wrapper_elsz];

			SparsePair * pair;
			for(int i=0;i<n_cols;i++){
				if(i==n_cols-1){
					n = n_nzelems - (p_cols[i]-1);
				}else{
					n = p_cols[i+1] - p_cols[i];
				}
				//std::cout << "`" << i << "    " << n << std::endl;
				for(int j=0;j<n;j++){
					int irow = rowpointers[p_rows[p_cols[i]-1+j]-1];
					pair = (SparsePair*) &_sparse_buffer_row[irow*wrapper_elsz];
					pair->idx = i+1;
					memcpy(&pair->data, &p_data[(p_cols[i]-1+j)*elsz], sizeof(elsz));
					rowpointers[p_rows[p_cols[i]-1+j]-1] ++;
					//assert(pair->idx != 0);
				}
			}

			/*
			SparsePair * pair;
			for(int i=0;i<n_rows;i++){
				int start = p_rows[i]-1;
				int end = n_nzelems;
				if(i!= n_rows-1) end = p_rows[i+1]-1;
				for(long j=start;j<end;j++){
					pair = (SparsePair*) &_sparse_buffer_row[j*wrapper_elsz];
					pair->idx = p_cols[j];
					memcpy(&pair->data, &p_data[j*elsz], sizeof(elsz));
				}
			}*/

			for(int i=0;i<n_rows;i++){
				//row_pointers[i].type = wrapped_el_type;
				row_pointers[i].data = (void*) &(_sparse_buffer_row[ rowpointers_ori[i]*wrapper_elsz ]);
				if(i==n_rows - 1){
					row_pointers[i].length = n_nzelems - rowpointers_ori[i];
				}else{
					row_pointers[i].length = rowpointers_ori[i+1] - rowpointers_ori[i];
				}
				row_pointers[i].elsize = wrapper_elsz;
				row_pointers[i].ptrarray = false;
				row_pointers[i].ndims = 1;
				row_pointers[i].isshared = 1;
				row_pointers[i].isaligned = 0;
				row_pointers[i].how = 0;
				row_pointers[i].nrows = row_pointers[i].length;
				row_pointers[i].maxsize = row_pointers[i].length;
				row_pointers[i].offset = 0;
				//row_pointers[i] = DenseVector<A>(p_data[i], n_cols);
			}
			task_row.row_pointers = row_pointers;
			task_row.shared_data = shared_data;
			dw_row_runner.prepare();
		}

	}


	/**
	 * \brief Register a row-access function.
	 *
	 * \return function handle that can used later to call this function.
	 */
	unsigned int register_row(
		double (* f) (const jl_array_t* const p_row, B* const p_model, jl_array_t *)
	){	
		fs_row[current_handle_id] = f;
		return current_handle_id ++;
	}

	/**
	 * \brief Register a column-access function.
	 *
	 * \return function handle that can used later to call this function.
	 */
	unsigned int register_col(
		double (* f) (const jl_array_t* const p_col, B* const p_model, jl_array_t *)
	){
		fs_col[current_handle_id] = f;
		return current_handle_id ++;
	}

	/**
	 * \brief Register a column-to-row-access function.
	 *
	 * \return function handle that can used later to call this function.
	 */
	unsigned int register_c2r(
		double (* f) (const jl_array_t* const p_col, int i_col,
					const jl_array_t* const p_rows,
					B* const p_model, jl_array_t *)
	){
		fs_c2r[current_handle_id] = f;
		return current_handle_id ++;
	}

	/**
	 * \brief Register a model averaging function.
	 *
	 * \param f_handle The function handle of a
	 * row-access/column-access/column-to-row-access function that
	 * this model averaging function will be used with.
	 */
	void register_model_avg(unsigned int f_handle, 
		void (* f) (jl_array_t* const p_models, int nreplicas, int ireplica)){
		fs_avg[f_handle] = f;
	}

	/**
	 * \brief Execute the function with a given handle.
	 *
	 * \return The function for each handle returns a double value
	 * after processing each row/column, this exec function will
	 * return a sum of these. This can be used, say, to calculate 
	 * the loss.  
	 */
	double exec(unsigned int f_handle){

		double rs = 0.0;

		Timer t;

		if(access_mode == DW_ACCESS_ROW){
			const DW_FUNCTION_ROW f = fs_row.find(f_handle)->second;
			DW_FUNCTION_MAVG f_avg = NULL;
			if(fs_avg.find(f_handle) != fs_avg.end()){
				f_avg = fs_avg.find(f_handle)->second;	
			}
			task_row.f = f;
			dw_row_runner.isjulia = true;
			rs = dw_row_runner.exec(row_ids, n_rows, dense_map_row_JULIA<B>, NULL, f_avg);
		}else if(access_mode == DW_ACCESS_COL){
			const DW_FUNCTION_COL f = fs_col.find(f_handle)->second;
			DW_FUNCTION_MAVG f_avg = NULL;
			if(fs_avg.find(f_handle) != fs_avg.end()){
				f_avg = fs_avg.find(f_handle)->second;	
			}
			task_col.f = f;
			dw_col_runner.isjulia = true;
			rs = dw_col_runner.exec(col_ids, n_cols, dense_map_col_JULIA<B>, NULL, f_avg);
		}else if(access_mode == DW_ACCESS_C2R){
			if(fs_row.find(f_handle) != fs_row.end()){
				const DW_FUNCTION_ROW  f = fs_row.find(f_handle)->second;
				DW_FUNCTION_MAVG f_avg = NULL;
				if(fs_avg.find(f_handle) != fs_avg.end()){
					f_avg = fs_avg.find(f_handle)->second;	
				}
				task_row.f = f;
				dw_row_runner.isjulia = true;
				rs = dw_row_runner.exec(row_ids, n_rows, dense_map_row_JULIA<B>, NULL, f_avg);
			}else{
				const DW_FUNCTION_C2R f = fs_c2r.find(f_handle)->second;
				DW_FUNCTION_MAVG f_avg = NULL;
				if(fs_avg.find(f_handle) != fs_avg.end()){
					f_avg = fs_avg.find(f_handle)->second;	
				}
				task_c2r.f = f;
				dw_c2r_runner.isjulia = true;
				rs = dw_c2r_runner.exec(col_ids, n_cols, dense_map_c2r_JULIA<B>, NULL, f_avg);
			}
		}

	    double data_byte = 1.0 * sizeof(double) * n_rows * n_cols;
	    double te = t.elapsed();
	    double throughput_gb = data_byte / te / 1024 / 1024 / 1024;
	    std::cout << "TIME=" << te << " secs" << " THROUGHPUT=" << throughput_gb << " GB/sec." << std::endl;

		return rs;
	}

};

#endif

