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


#ifndef _DENSE_DW_H
#define _DENSE_DW_H

#include "util.h"
#include "dimmwitted_const.h"
#include "dimmwitted_dstruct.h"

#include "engine/scheduler.h"
#include "engine/scheduler_strawman.h"
#include "engine/scheduler_hogwild.h"
#include "engine/scheduler_percore.h"
#include "engine/scheduler_pernode.h"

/**
 * \brief Task description for the row-access method.
 *
 * \tparam A type of data elements (See DenseDimmWitted)
 * \tparam B type of model (See DenseDimmWitted)
 */
template<class A, class B>
class TASK_ROW{
public:
	DenseVector<A>* row_pointers; /**<A list of pointer to rows*/
	double (*f) (const DenseVector<A>* const, B* const);	/**<Row-access function to execute*/
	TASK_ROW(DenseVector<A>* const _row_pointers, 
			double (* _f) (const DenseVector<A>* const p_row, B* const p_model)):
		row_pointers(_row_pointers), f(_f)
	{}
	TASK_ROW(){}
};

/**
 * \brief Task description for the column-access method.
 *
 * \tparam A type of data elements (See DenseDimmWitted)
 * \tparam B type of model (See DenseDimmWitted)
 */
template<class A, class B>
class TASK_COL{
public:
	DenseVector<A>* col_pointers; /**<A list of pointer to columns*/
	double (*f) (const DenseVector<A>* const, B* const);	/**<Column-access function to execute*/
	TASK_COL(DenseVector<A>* const _col_pointers,
		double (*_f) (const DenseVector<A>* const, B* const)):
		col_pointers(_col_pointers), f(_f)
	{}
	TASK_COL(){}
};

/**
 * \brief Task description for the column-to-row-access method.
 *
 * \tparam A type of data elements (See DenseDimmWitted)
 * \tparam B type of model (See DenseDimmWitted)
 */
template<class A, class B>
class TASK_C2R{
public:
	DenseVector<A>*  col_pointers; /**<A list of pointer to columns*/
	Pair<long, long> *  c2r_col_2_rowbuffer_idxs; /**<See DenseDimmWitted::c2r_col_2_rowbuffer_idxs*/
	DenseVector<A>*  c2r_row_pointers_buffer; /**<See DenseDimmWitted::_c2r_row_pointers_buffer*/
	double (*f) (const DenseVector<A>* const p_col, int i_col,
																		 const DenseVector<A>* const p_rows, int, B* const);
																	/**<Column-to-row-access function to execute*/

	TASK_C2R(DenseVector<A>* _col_pointers, Pair<long, long> * _c2r_col_2_rowbuffer_idxs,
		DenseVector<A>* _c2r_row_pointers_buffer, 
		double (*_f) (const DenseVector<A>* const p_col, int i_col,
																		 const DenseVector<A>* const p_rows, int, B* const)):
		col_pointers(_col_pointers), c2r_col_2_rowbuffer_idxs(_c2r_col_2_rowbuffer_idxs),
		c2r_row_pointers_buffer(_c2r_row_pointers_buffer), f(_f)
	{}

	TASK_C2R(){}

};

/**
 * \brief A thin wrapper for the row-access function that can be used by DWRun (engine/scheduler.h).
 *
 * \tparam A type of data elements (See DenseDimmWitted)
 * \tparam B type of model (See DenseDimmWitted)
 */
template<class A, class B>
double dense_map_row (long i_task, const TASK_ROW<A,B> * const rddata, B * const wrdata){
	return rddata->f(&rddata->row_pointers[i_task], wrdata);
}

/**
 * \brief A thin wrapper for the column-access function that can be used by DWRun (engine/scheduler.h).
 *
 * \tparam A type of data elements (See DenseDimmWitted)
 * \tparam B type of model (See DenseDimmWitted)
 */
template<class A, class B>
double dense_map_col (long i_task, const TASK_COL<A,B> * const rddata, B * const wrdata){
	return rddata->f(&rddata->col_pointers[i_task], wrdata);
}

/**
 * \brief A thin wrapper for the column-to-row-access function that can be used by DWRun (engine/scheduler.h).
 *
 * \tparam A type of data elements (See DenseDimmWitted)
 * \tparam B type of model (See DenseDimmWitted)
 */
template<class A, class B>
double dense_map_c2r (long i_task, const TASK_C2R<A,B> * const rddata, B * const wrdata){
	const Pair<long, long> & row_ptrs = rddata->c2r_col_2_rowbuffer_idxs[i_task];
	return rddata->f(&rddata->col_pointers[i_task], i_task, &(rddata->c2r_row_pointers_buffer[row_ptrs.first]), 
						row_ptrs.second, wrdata);
}

/**
 * \brief A thin wrapper for allocating model replicas that can be used by DWRun (engine/scheduler.h).
 *
 * \tparam A type of data elements (See DenseDimmWitted)
 * \tparam B type of model (See DenseDimmWitted)
 */
template<class A, class B>
void dense_model_allocator (B ** const a, const B * const b){
  *a = new B(*b);
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
template<class A, class B, ModelReplType model_repl_type, DataReplType data_repl_type, AccessMode access_mode>
class DenseDimmWitted{
public:

	typedef double (*DW_FUNCTION_ROW) (const DenseVector<A>* const, B* const);
		/**<\brief Type of row-access function*/

	typedef double (*DW_FUNCTION_COL) (const DenseVector<A>* const, B* const);
		/**<\brief Type of column-access function*/

	typedef double (*DW_FUNCTION_C2R) (const DenseVector<A>* const p_col, int i_col,
																		 const DenseVector<A>* const p_rows, int, B* const);
		/**<\brief Type of column-to-row-access function*/

	typedef void (*DW_FUNCTION_MAVG) (B** const p_models, int nreplicas, int ireplica);
		/**<\brief Type of the function that conducts model averaging*/

	A** const p_data; /**<\brief Pointer to the data, which is a two dimensional array of type A*/

	long n_rows; /**<\brief Number of rows of the data.*/

	long n_cols; /**<\brief Number of columns of the data.*/
	
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

	DenseVector<A>* const row_pointers; /**<\brief Pointer to each row of the data.*/

	DenseVector<A>* const col_pointers; /**<\brief Pointer to each column of the data.*/

	DenseVector<A>* _c2r_row_pointers_buffer; /**<\brief For column-to-row-access, this data structure
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

	Pair<long, long> * const c2r_col_2_rowbuffer_idxs; /**<\brief For For column-to-row-access, this data structure
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

	A* _new_ele_buffer; /**<\brief If the input is row-wise storage, and the user requires column-wise
                         access, this pointer points to the re-structured region of
                         the memory for column-wise storage*/

	long * row_ids; /**<\brief A list of indexes that the system will access. For row-access, it will be
                  [0,nrows]. */

	long * col_ids; /**<\brief A list of indexes that the system will access. For column-access or 
									 column-to-row access, it will be [0,ncols]. */

	TASK_ROW<A, B> task_row; /**<\brief Task description for row-access. */

	TASK_COL<A, B> task_col; /**<\brief Task description for column-access. */

	TASK_C2R<A, B> task_c2r; /**<\brief Task description for column-to-row access. */

	DWRun<TASK_ROW<A,B>, B, model_repl_type, data_repl_type> dw_row_runner;
		/**<\brief Execution backend for row-access. See engine/scheduler.h. */

	DWRun<TASK_COL<A,B>, B, model_repl_type, data_repl_type> dw_col_runner;
		/**<\brief Execution backend for column-access. See engine/scheduler.h. */

	DWRun<TASK_C2R<A,B>, B, model_repl_type, data_repl_type> dw_c2r_runner;
		/**<\brief Execution backend for column-to-row-access. See engine/scheduler.h. */

public:

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
	 * \brief Constructor of dense DimmWitted.
	 *
	 * \param _data The data as a row-wise stored two dimensional array of type A.
	 * \param _n_rows Number of rows in the data.
	 * \param _n_cols Number of columns in the data.
	 * \param _model Pointer to the model.
	 */
	DenseDimmWitted(A** _data, long _n_rows, long _n_cols, B * const _model
	):
		p_data(_data), p_model(_model),
		n_rows(_n_rows), n_cols(_n_cols),
		current_handle_id(0),
		row_pointers((DenseVector<A>*) ::operator new(_n_rows * sizeof(DenseVector<A>))),
		col_pointers((DenseVector<A>*) ::operator new(_n_cols * sizeof(DenseVector<A>))),
		c2r_col_2_rowbuffer_idxs(new Pair<long, long>[_n_cols]),
		row_ids(new long[_n_rows]),
		col_ids(new long[_n_cols]),
		dw_row_runner(DWRun<TASK_ROW<A,B>, B, model_repl_type, data_repl_type>(&task_row, p_model, dense_model_allocator<A,B>)),
		dw_col_runner(DWRun<TASK_COL<A,B>, B, model_repl_type, data_repl_type>(&task_col, p_model, dense_model_allocator<A,B>)),
		dw_c2r_runner(DWRun<TASK_C2R<A,B>, B, model_repl_type, data_repl_type>(&task_c2r, p_model, dense_model_allocator<A,B>))
	{
		for(int i=0;i<n_rows;i++){
			row_ids[i] = i;
		}
		for(int j=0;j<n_cols;j++){
			col_ids[j] = j;
		}

		if(access_mode == DW_ACCESS_ROW){
			for(int i=0;i<n_rows;i++){
				row_pointers[i] = DenseVector<A>(p_data[i], n_cols);
			}
			task_row.row_pointers = row_pointers;
			dw_row_runner.prepare();

		}else{
			_new_ele_buffer = new A[n_rows*n_cols];
			long ct = 0;
			for(int j=0;j<n_cols;j++){
				col_pointers[j] = DenseVector<A>(&_new_ele_buffer[ct], n_rows);
				for(int i=0;i<n_rows;i++){
					_new_ele_buffer[ct++] = _data[i][j];
				}
			}
			task_col.col_pointers = col_pointers;
			dw_col_runner.prepare();

			if(access_mode == DW_ACCESS_C2R){

				for(int i=0;i<n_rows;i++){
					row_pointers[i] = DenseVector<A>(p_data[i], n_cols);
				}

				_c2r_row_pointers_buffer = (DenseVector<A>*) ::operator new(n_rows * n_cols * sizeof(DenseVector<A>));
				long ct = 0;
				for(int j=0;j<n_cols;j++){
					c2r_col_2_rowbuffer_idxs[j].first = ct;
					int a = 0;
					for(int i=0;i<n_rows;i++){
						if(col_pointers[j].p[i] != 0){
							a++;
							_c2r_row_pointers_buffer[ct].p = row_pointers[i].p;
							_c2r_row_pointers_buffer[ct].n = row_pointers[i].n;

							ct ++;
						}
					}
					c2r_col_2_rowbuffer_idxs[j].second = a;
				}

				task_row.row_pointers = row_pointers;
				dw_row_runner.prepare();

				task_c2r.col_pointers = col_pointers;
				task_c2r.c2r_col_2_rowbuffer_idxs = c2r_col_2_rowbuffer_idxs;
				task_c2r.c2r_row_pointers_buffer = _c2r_row_pointers_buffer;
				dw_c2r_runner.prepare();

			}
		}
	}

	/**
	 * \brief Register a row-access function.
	 *
	 * \return function handle that can used later to call this function.
	 */
	unsigned int register_row(
		double (* f) (const DenseVector<A>* const p_row, B* const p_model)
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
		double (* f) (const DenseVector<A>* const p_col, int n_row, B* const p_model)
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
		double (* f) (const DenseVector<A>* const p_col, int i_col,
					const DenseVector<A>* const p_rows, int n_rows,
					B* const p_model)
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
		void (* f) (B** const p_models, int nreplicas, int ireplica)){
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

		if(access_mode == DW_ACCESS_ROW){
			const DW_FUNCTION_ROW f = fs_row.find(f_handle)->second;
			DW_FUNCTION_MAVG f_avg = NULL;
			if(fs_avg.find(f_handle) != fs_avg.end()){
				f_avg = fs_avg.find(f_handle)->second;	
			}
			task_row.f = f;
			rs = dw_row_runner.exec(row_ids, n_rows, dense_map_row<A,B>, f_avg, NULL);
		}else if(access_mode == DW_ACCESS_COL){
			const DW_FUNCTION_COL f = fs_col.find(f_handle)->second;
			DW_FUNCTION_MAVG f_avg = NULL;
			if(fs_avg.find(f_handle) != fs_avg.end()){
				f_avg = fs_avg.find(f_handle)->second;	
			}
			task_col.f = f;
			rs = dw_col_runner.exec(col_ids, n_cols, dense_map_col<A,B>, f_avg, NULL);
		}else if(access_mode == DW_ACCESS_C2R){
			if(fs_row.find(f_handle) != fs_row.end()){
				const DW_FUNCTION_ROW  f = fs_row.find(f_handle)->second;
				DW_FUNCTION_MAVG f_avg = NULL;
				if(fs_avg.find(f_handle) != fs_avg.end()){
					f_avg = fs_avg.find(f_handle)->second;	
				}
				task_row.f = f;
				rs = dw_row_runner.exec(row_ids, n_rows, dense_map_row<A,B>, f_avg, NULL);
			}else{
				const DW_FUNCTION_C2R f = fs_c2r.find(f_handle)->second;
				DW_FUNCTION_MAVG f_avg = NULL;
				if(fs_avg.find(f_handle) != fs_avg.end()){
					f_avg = fs_avg.find(f_handle)->second;	
				}
				task_c2r.f = f;
				rs = dw_c2r_runner.exec(col_ids, n_cols, dense_map_c2r<A,B>, f_avg, NULL);
			}
		}

		return rs;
	}

};

#endif

