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


#ifndef _SPARSE_DW_H
#define _SPARSE_DW_H

#include "util.h"
#include "dimmwitted_const.h"
#include "dimmwitted_dstruct.h"

#include "engine/scheduler.h"
#include "engine/scheduler_strawman.h"
#include "engine/scheduler_hogwild.h"
#include "engine/scheduler_percore.h"
#include "engine/scheduler_pernode.h"

template<class A, class B>
class TASK_ROW_SPARSE{
public:
	SparseVector<A>* row_pointers; 
	double (*f) (const SparseVector<A>* const, B* const);
	TASK_ROW_SPARSE(SparseVector<A>* const _row_pointers, 
			double (* _f) (const SparseVector<A>* const p_row, B* const p_model)):
		row_pointers(_row_pointers), f(_f)
	{}
	TASK_ROW_SPARSE(){}
};

template<class A, class B>
class TASK_COL_SPARSE{
public:
	SparseVector<A>* col_pointers; 
	double (*f) (const SparseVector<A>* const, B* const);
	TASK_COL_SPARSE(SparseVector<A>* const _col_pointers,
		double (*_f) (const SparseVector<A>* const, B* const)):
		col_pointers(_col_pointers), f(_f)
	{}
	TASK_COL_SPARSE(){}
};

template<class A, class B>
class TASK_C2R_SPARSE{
public:
	SparseVector<A>*  col_pointers; 
	Pair<long, long> *  c2r_col_2_rowbuffer_idxs;
	SparseVector<A>*  c2r_row_pointers_buffer;
	double (*f) (const SparseVector<A>* const p_col, int i_col,
										const SparseVector<A>* const p_rows, int, B* const);

	TASK_C2R_SPARSE(SparseVector<A>* _col_pointers, Pair<long, long> * _c2r_col_2_rowbuffer_idxs,
		SparseVector<A>* _c2r_row_pointers_buffer,
		double (*_f) (const SparseVector<A>* const p_col, int i_col,
										const SparseVector<A>* const p_rows, int, B* const)):
		col_pointers(_col_pointers), c2r_col_2_rowbuffer_idxs(_c2r_col_2_rowbuffer_idxs),
		c2r_row_pointers_buffer(_c2r_row_pointers_buffer), f(_f)
	{}

	TASK_C2R_SPARSE(){}

};

template<class A, class B>
double sparse_map_row (long i_task, const TASK_ROW_SPARSE<A,B> * const rddata, B * const wrdata){
	return rddata->f(&rddata->row_pointers[i_task], wrdata);
}

template<class A, class B>
double sparse_map_col (long i_task, const TASK_COL_SPARSE<A,B> * const rddata, B * const wrdata){
	return rddata->f(&rddata->col_pointers[i_task], wrdata);
}

template<class A, class B>
double sparse_map_c2r (long i_task, const TASK_C2R_SPARSE<A,B> * const rddata, B * const wrdata){
	const Pair<long, long> & row_ptrs = rddata->c2r_col_2_rowbuffer_idxs[i_task];
	return rddata->f(&rddata->col_pointers[i_task], i_task, &(rddata->c2r_row_pointers_buffer[row_ptrs.first]), 
						row_ptrs.second, wrdata);
}

template<class A, class B>
void sparse_model_allocator (B ** const a, const B * const b){
  *a = new B(*b);
}

/**
 * \brief This class is the interface of DimmWitted for Sparse data.
 * Detailed documentation of functions that are similar
 * as the Dense data can be found in DenseDimmWitted.
 *
 * \tparam A Type of elements for data.
 * \tparam B Type of model.
 * \tparam model_repl_type Model replication strategy.
 * \tparam data_repl_type Data replication strategy.
 * \tparam access_mode Access Method
 */
template<class A, class B, ModelReplType model_repl_type, DataReplType data_repl_type, AccessMode access_mode>
class SparseDimmWitted{

	typedef double (*DW_FUNCTION_ROW) (const SparseVector<A>* const, B* const);

	typedef double (*DW_FUNCTION_COL) (const SparseVector<A>* const, B* const);

	typedef double (*DW_FUNCTION_C2R) (const SparseVector<A>* const p_col, int i_col,
												 const SparseVector<A>* const p_rows, int, B* const);

	typedef void (*DW_FUNCTION_MAVG) (B** const p_models, int nreplicas, int ireplica);

	A* p_data;

	long n_rows;

	long n_cols;

	long n_elems;

	B* p_model;

	std::map<unsigned int, DW_FUNCTION_ROW> fs_row;

	std::map<unsigned int, DW_FUNCTION_COL> fs_col;

	std::map<unsigned int, DW_FUNCTION_C2R> fs_c2r;

	std::map<unsigned int, DW_FUNCTION_MAVG> fs_avg;

	unsigned int current_handle_id;

	SparseVector<A>* const row_pointers; 

	SparseVector<A>* const col_pointers;

	Pair<long, long> * const c2r_col_2_rowbuffer_idxs;

	SparseVector<A>* _c2r_row_pointers_buffer;

	A* _new_ele_buffer;

	long * tasks;

	long * row_ids;

	long * col_ids;

	TASK_ROW_SPARSE<A, B> task_row;

	TASK_COL_SPARSE<A, B> task_col;

	TASK_C2R_SPARSE<A, B> task_c2r;

	DWRun<TASK_ROW_SPARSE<A,B>, B, model_repl_type, data_repl_type> dw_row_runner;

	DWRun<TASK_COL_SPARSE<A,B>, B, model_repl_type, data_repl_type> dw_col_runner;

	DWRun<TASK_C2R_SPARSE<A,B>, B, model_repl_type, data_repl_type> dw_c2r_runner;

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
	 * \brief Constructor of dense DimmWitted. The input data
	 * (_data, rows, cols) is assumed to be stored in the
	 * CSR format, _n_elems is the number of non-zero
	 * elements.
	 */
	SparseDimmWitted(
		SparseVector<A>* _row_pointers, long _n_rows, long _n_cols, long _n_elems, 
		B * const _model):
		p_data(_row_pointers[0].p),
		p_model(_model),
		n_rows(_n_rows), n_cols(_n_cols), n_elems(_n_elems),
		current_handle_id(0),
		row_pointers(_row_pointers),
		col_pointers((SparseVector<A>*) ::operator new(_n_cols * sizeof(SparseVector<A>))),
		c2r_col_2_rowbuffer_idxs(new Pair<long, long>[_n_cols]),
		row_ids(new long[_n_rows]),
		col_ids(new long[_n_cols]),
		dw_row_runner(DWRun<TASK_ROW_SPARSE<A,B>, B, model_repl_type, data_repl_type>(&task_row, p_model, sparse_model_allocator<A,B>)),
		dw_col_runner(DWRun<TASK_COL_SPARSE<A,B>, B, model_repl_type, data_repl_type>(&task_col, p_model, sparse_model_allocator<A,B>)),
		dw_c2r_runner(DWRun<TASK_C2R_SPARSE<A,B>, B, model_repl_type, data_repl_type>(&task_c2r, p_model, sparse_model_allocator<A,B>))
	{
		for(int i=0;i<n_rows;i++){
			row_ids[i] = i;
		}
		for(int j=0;j<n_cols;j++){
			col_ids[j] = j;
		}

		if(access_mode != DW_ACCESS_ROW){
			assert(false && "This constructor only support row-access right now.");
		}

		task_row.row_pointers = row_pointers;
		dw_row_runner.prepare();

	}

	/**
	 * \brief Constructor of dense DimmWitted. The input data
	 * (_data, rows, cols) is assumed to be stored in the
	 * CSR format, _n_elems is the number of non-zero
	 * elements.
	 */
	SparseDimmWitted(
		A* _data, long * rows, long * cols, long _n_rows, long _n_cols, long _n_elems, 
		B * const _model):
		p_data(_data), p_model(_model),
		n_rows(_n_rows), n_cols(_n_cols), n_elems(_n_elems),
		current_handle_id(0),
		row_pointers((SparseVector<A>*) ::operator new(_n_rows * sizeof(SparseVector<A>))),
		col_pointers((SparseVector<A>*) ::operator new(_n_cols * sizeof(SparseVector<A>))),
		c2r_col_2_rowbuffer_idxs(new Pair<long, long>[_n_cols]),
		row_ids(new long[_n_rows]),
		col_ids(new long[_n_cols]),
		dw_row_runner(DWRun<TASK_ROW_SPARSE<A,B>, B, model_repl_type, data_repl_type>(&task_row, p_model, sparse_model_allocator<A,B>)),
		dw_col_runner(DWRun<TASK_COL_SPARSE<A,B>, B, model_repl_type, data_repl_type>(&task_col, p_model, sparse_model_allocator<A,B>)),
		dw_c2r_runner(DWRun<TASK_C2R_SPARSE<A,B>, B, model_repl_type, data_repl_type>(&task_c2r, p_model, sparse_model_allocator<A,B>))
	{

		for(int i=0;i<n_rows;i++){
			row_ids[i] = i;
		}
		for(int j=0;j<n_cols;j++){
			col_ids[j] = j;
		}

		if(access_mode == DW_ACCESS_ROW || access_mode == DW_ACCESS_C2R){
			for(int i=0;i<n_rows;i++){
				if(i==n_rows-1){
					row_pointers[i] = SparseVector<A>(&p_data[rows[i]], &cols[rows[i]], n_elems - rows[i]);
				}else{
					row_pointers[i] = SparseVector<A>(&p_data[rows[i]], &cols[rows[i]], rows[i+1] - rows[i]);
				}
			}
			task_row.row_pointers = row_pointers;
			dw_row_runner.prepare();

		}

		if(access_mode == DW_ACCESS_COL || access_mode == DW_ACCESS_C2R){
			_new_ele_buffer = new A[n_elems];
			long * colcounts = new long[n_cols];
			long * colpointers = new long[n_cols];
			long * colpointers_ori = new long[n_cols];
			long * colrows = new long[n_elems];
			for(int i=0;i<n_cols;i++){
				colcounts[i] = 0;
			}
			int n = 0;
			for(int i=0;i<n_rows;i++){
				if(i==n_rows-1){
					n = n_elems - rows[i];
				}else{
					n = rows[i+1] - rows[i];
				}
				for(int j=0;j<n;j++){
					colcounts[cols[rows[i]+j]] ++;
				}
			}

			for(int i=0;i<n_cols;i++){
				colpointers[i] = 0;
			}

			for(int i=1;i<n_cols;i++){
				colpointers[i] = colpointers[i-1] + colcounts[i-1];
				colpointers_ori[i] = colpointers[i];
			}

			long sum = 0;
			for(int i=0;i<n_cols;i++){
				sum += colcounts[i];
			}

			for(int i=0;i<n_rows;i++){
				if(i==n_rows-1){
					n = n_elems - rows[i];
				}else{
					n = rows[i+1] - rows[i];
				}
				for(int j=0;j<n;j++){
					int col = cols[rows[i]+j];
					int icol = colpointers[j];
					colrows[icol] = i;
					_new_ele_buffer[icol] = p_data[rows[i]+j];
					colpointers[j] ++;
				}
			}

			for(int i=0;i<n_cols;i++){
				if(i==n_cols-1){
					col_pointers[i] = SparseVector<A>(&_new_ele_buffer[colpointers_ori[i]], &colrows[colpointers_ori[i]], n_elems - colpointers_ori[i]);
				}else{
					col_pointers[i] = SparseVector<A>(&_new_ele_buffer[colpointers_ori[i]], &colrows[colpointers_ori[i]], colpointers_ori[i+1] - colpointers_ori[i]);
				}
			}

			task_col.col_pointers = col_pointers;
			dw_col_runner.prepare();

			if(access_mode == DW_ACCESS_C2R){

				long ct = 0;
				for(int j=0;j<n_cols;j++){
					c2r_col_2_rowbuffer_idxs[j].first = ct;
					int a = 0;
					for(int i=0;i<col_pointers[j].n;i++){
						a++;
						_c2r_row_pointers_buffer[ct++] = row_pointers[col_pointers[j].idxs[i]];
					}
					c2r_col_2_rowbuffer_idxs[j].second = a;
				}

				task_c2r.col_pointers = col_pointers;
				task_c2r.c2r_col_2_rowbuffer_idxs = c2r_col_2_rowbuffer_idxs;
				task_c2r.c2r_row_pointers_buffer = _c2r_row_pointers_buffer;
				dw_c2r_runner.prepare();

			}
			
		}
	}

	void register_model_avg(unsigned int f_handle, 
		void (* f) (B** const p_models, int nreplicas, int ireplica)){
		fs_avg[f_handle] = f;
	}

	unsigned int register_row(
		double (* f) (const SparseVector<A>* const p_row, B* const p_model)
	){	
		fs_row[current_handle_id] = f;
		return current_handle_id ++;
	}

	unsigned int register_col(
		double (* f) (const SparseVector<A>* const p_col, int n_row, B* const p_model)
	){
		fs_col[current_handle_id] = f;
		return current_handle_id ++;
	}

	unsigned int register_c2r(
		double (* f) (const SparseVector<A>* const p_col, int i_col,
					const SparseVector<A>* const p_rows, int n_rows,
					B* const p_model)
	){
		fs_c2r[current_handle_id] = f;
		return current_handle_id ++;
	}

	double dump_row(unsigned int f_handle, std::string filename){
		assert(access_mode == DW_ACCESS_ROW);
		const DW_FUNCTION_ROW f = fs_row.find(f_handle)->second;
		DW_FUNCTION_MAVG f_avg = NULL;
		if(fs_avg.find(f_handle) != fs_avg.end()){
			f_avg = *fs_avg.find(f_handle)->second;	
		}
		task_row.f = f;
		dw_row_runner.dump_row(row_ids, n_rows, sparse_map_row<A,B>, f_avg, NULL, filename);
		return 0;
	}

	double exec(unsigned int f_handle){

		double data_byte = 1.0 * sizeof(A) * n_elems + sizeof(long) * n_elems + sizeof(long) * n_rows;
		Timer t;

		double rs = 0.0;

		if(access_mode == DW_ACCESS_ROW){

			const DW_FUNCTION_ROW f = fs_row.find(f_handle)->second;
			DW_FUNCTION_MAVG f_avg = NULL;
			if(fs_avg.find(f_handle) != fs_avg.end()){
				f_avg = *fs_avg.find(f_handle)->second;	
			}
			task_row.f = *f;
			rs = dw_row_runner.exec(row_ids, n_rows, sparse_map_row<A,B>, f_avg, NULL);

		}else if(access_mode == DW_ACCESS_COL){
			const DW_FUNCTION_COL f = fs_col.find(f_handle)->second;
			DW_FUNCTION_MAVG  f_avg = NULL;
			if(fs_avg.find(f_handle) != fs_avg.end()){
				f_avg = *fs_avg.find(f_handle)->second;	
			}
			task_col.f = *f;
			rs = dw_col_runner.exec(col_ids, n_cols, sparse_map_col<A,B>, f_avg, NULL);

		}else if(access_mode == DW_ACCESS_C2R){
			if(fs_row.find(f_handle) != fs_row.end()){
				const DW_FUNCTION_ROW f = fs_row.find(f_handle)->second;
				DW_FUNCTION_MAVG  f_avg = NULL;
				if(fs_avg.find(f_handle) != fs_avg.end()){
					f_avg = *fs_avg.find(f_handle)->second;	
				}
				task_row.f = *f;
				rs = dw_row_runner.exec(row_ids, n_rows, sparse_map_row<A,B>, f_avg, NULL);

			}else{
				const DW_FUNCTION_C2R f = fs_c2r.find(f_handle)->second;
				DW_FUNCTION_MAVG  f_avg = NULL;
				if(fs_avg.find(f_handle) != fs_avg.end()){
					f_avg = *fs_avg.find(f_handle)->second;	
				}
				task_c2r.f = *f;
				rs = dw_c2r_runner.exec(col_ids, n_cols, sparse_map_c2r<A,B>, f_avg, NULL);

			}
		}

		double te = t.elapsed();
		double throughput_gb = data_byte / te / 1024 / 1024 / 1024;
		std::cout.precision(3);
		std::cout << "[DimmWitted FUNC=" << f_handle << "] " 
				  << "TIME=" << std::setw(6) << te << " secs"
				  << " THROUGHPUT=" << std::setw(6) << throughput_gb << " GB/sec." << std::endl;
		
		return rs;
	}

};

#endif

