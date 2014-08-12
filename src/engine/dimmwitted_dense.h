/**
Copyright 2014 Hazy Research (http://i.stanford.edu/hazy)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
**/

#ifndef _DENSE_DW_H
#define _DENSE_DW_H

#include "common.h"

#include "engine/scheduler.h"
#include "engine/scheduler_strawman.h"
#include "engine/scheduler_hogwild.h"
#include "engine/scheduler_percore.h"
#include "engine/scheduler_pernode.h"

template<class A, class B>
class TASK_ROW{
public:
	DenseVector<A>* row_pointers; 
	double (*f) (const DenseVector<A>* const, B* const);
	TASK_ROW(DenseVector<A>* const _row_pointers, 
			double (* _f) (const DenseVector<A>* const p_row, B* const p_model)):
		row_pointers(_row_pointers), f(_f)
	{}
	TASK_ROW(){}
};

template<class A, class B>
class TASK_COL{
public:
	DenseVector<A>* col_pointers; 
	double (*f) (const DenseVector<A>* const, B* const);
	TASK_COL(DenseVector<A>* const _col_pointers,
		double (*_f) (const DenseVector<A>* const, B* const)):
		col_pointers(_col_pointers), f(_f)
	{}
	TASK_COL(){}
};

template<class A, class B>
class TASK_C2R{
public:
	DenseVector<A>*  col_pointers; 
	Pair<long, long> *  c2r_col_2_rowbuffer_idxs;
	DenseVector<A>*  c2r_row_pointers_buffer;
	double (*f) (const DenseVector<A>* const p_col, int i_col,
																		 const DenseVector<A>* const p_rows, int, B* const);

	TASK_C2R(DenseVector<A>* _col_pointers, Pair<long, long> * _c2r_col_2_rowbuffer_idxs,
		DenseVector<A>* _c2r_row_pointers_buffer, 
		double (*_f) (const DenseVector<A>* const p_col, int i_col,
																		 const DenseVector<A>* const p_rows, int, B* const)):
		col_pointers(_col_pointers), c2r_col_2_rowbuffer_idxs(_c2r_col_2_rowbuffer_idxs),
		c2r_row_pointers_buffer(_c2r_row_pointers_buffer), f(_f)
	{}

	TASK_C2R(){}

};

template<class A, class B>
double dense_map_row (long i_task, const TASK_ROW<A,B> * const rddata, B * const wrdata){
	return rddata->f(&rddata->row_pointers[i_task], wrdata);
}

template<class A, class B>
double dense_map_col (long i_task, const TASK_COL<A,B> * const rddata, B * const wrdata){
	return rddata->f(&rddata->col_pointers[i_task], wrdata);
}

template<class A, class B>
double dense_map_c2r (long i_task, const TASK_C2R<A,B> * const rddata, B * const wrdata){
	const Pair<long, long> & row_ptrs = rddata->c2r_col_2_rowbuffer_idxs[i_task];
	return rddata->f(&rddata->col_pointers[i_task], i_task, &(rddata->c2r_row_pointers_buffer[row_ptrs.first]), 
						row_ptrs.second, wrdata);
}

template<class A, class B>
void dense_model_allocator (B ** const a, const B * const b){
  *a = new B(*b);
}

/**
 * This class is the interface of 
 */
template<class A, class B, ModelReplType model_repl_type, DataReplType data_repl_type, AccessMode access_mode>
class DenseDimmWitted{

	typedef double (*DW_FUNCTION_ROW) (const DenseVector<A>* const, B* const);

	typedef double (*DW_FUNCTION_COL) (const DenseVector<A>* const, B* const);

	typedef double (*DW_FUNCTION_C2R) (const DenseVector<A>* const p_col, int i_col,
																		 const DenseVector<A>* const p_rows, int, B* const);

	typedef void (*DW_FUNCTION_MAVG) (B** const p_models, int nreplicas, int ireplica);

	A** const p_data;

	long n_rows;

	long n_cols;
	
	B* const p_model;

	std::map<unsigned int, DW_FUNCTION_ROW*> fs_row;

	std::map<unsigned int, DW_FUNCTION_COL*> fs_col;

	std::map<unsigned int, DW_FUNCTION_C2R*> fs_c2r;

	std::map<unsigned int, DW_FUNCTION_MAVG*> fs_avg;

	unsigned int current_handle_id;

	DenseVector<A>* const row_pointers; 

	DenseVector<A>* const col_pointers;

	Pair<long, long> * const c2r_col_2_rowbuffer_idxs;

	DenseVector<A>* _c2r_row_pointers_buffer;

	A* _new_ele_buffer;

	long * tasks;

	long * row_ids;

	long * col_ids;

	TASK_ROW<A, B> task_row;

	TASK_COL<A, B> task_col;

	TASK_C2R<A, B> task_c2r;

	DWRun<TASK_ROW<A,B>, B, model_repl_type, data_repl_type> dw_row_runner;

	DWRun<TASK_COL<A,B>, B, model_repl_type, data_repl_type> dw_col_runner;

	DWRun<TASK_C2R<A,B>, B, model_repl_type, data_repl_type> dw_c2r_runner;

public:

	DenseDimmWitted(A** _data, long _n_rows, long _n_cols,
	   B * const _model
	):
		p_data(_data), p_model(_model),
		n_rows(_n_rows), n_cols(_n_cols),
		current_handle_id(0),
		row_pointers((DenseVector<A>*) ::operator new(_n_rows * sizeof(DenseVector<A>))),
		col_pointers((DenseVector<A>*) ::operator new(_n_cols * sizeof(DenseVector<A>))),
		c2r_col_2_rowbuffer_idxs(new Pair<long, long>[_n_rows]),
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

		if(access_mode == DW_ROW){
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

			if(access_mode == DW_C2R){

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

	void register_model_avg(unsigned int f_handle, 
		void (* f) (B** const p_models, int nreplicas, int ireplica)){
		fs_avg[f_handle] = &f;
	}

	unsigned int register_row(
		double (* f) (const DenseVector<A>* const p_row, B* const p_model)
	){	
		fs_row[current_handle_id] = &f;
		return current_handle_id ++;
	}

	unsigned int register_col(
		double (* f) (const DenseVector<A>* const p_col, int n_row, B* const p_model)
	){
		fs_col[current_handle_id] = &f;
		return current_handle_id ++;
	}

	unsigned int register_c2r(
		double (* f) (const DenseVector<A>* const p_col, int i_col,
					const DenseVector<A>* const p_rows, int n_rows,
					B* const p_model)
	){
		fs_c2r[current_handle_id] = &f;
		return current_handle_id ++;
	}

	double exec(unsigned int f_handle){

		Timer t;
		double rs = 0.0;

		if(access_mode == DW_ROW){
			const DW_FUNCTION_ROW * const f = fs_row.find(f_handle)->second;
			DW_FUNCTION_MAVG f_avg = NULL;
			if(fs_avg.find(f_handle) != fs_avg.end()){
				f_avg = *fs_avg.find(f_handle)->second;	
			}
			task_row.f = *f;
			dw_row_runner.prepare();
			rs = dw_row_runner.exec(row_ids, n_rows, dense_map_row<A,B>, f_avg, NULL);
		}else if(access_mode == DW_COL){
			const DW_FUNCTION_COL * const f = fs_col.find(f_handle)->second;
			DW_FUNCTION_MAVG  f_avg = NULL;
			if(fs_avg.find(f_handle) != fs_avg.end()){
				f_avg = *fs_avg.find(f_handle)->second;	
			}
			task_col.f = *f;
			dw_col_runner.prepare();
			rs = dw_col_runner.exec(col_ids, n_cols, dense_map_col<A,B>, f_avg, NULL);
		}else if(access_mode == DW_C2R){
			if(fs_row.find(f_handle) != fs_row.end()){
				const DW_FUNCTION_ROW * const f = fs_row.find(f_handle)->second;
				DW_FUNCTION_MAVG  f_avg = NULL;
				if(fs_avg.find(f_handle) != fs_avg.end()){
					f_avg = *fs_avg.find(f_handle)->second;	
				}
				task_row.f = *f;
				rs = dw_row_runner.exec(row_ids, n_rows, dense_map_row<A,B>, f_avg, NULL);
			}else{
				const DW_FUNCTION_C2R * f = fs_c2r.find(f_handle)->second;
				DW_FUNCTION_MAVG  f_avg = NULL;
				if(fs_avg.find(f_handle) != fs_avg.end()){
					f_avg = *fs_avg.find(f_handle)->second;	
				}
				task_c2r.f = *f;
				rs = dw_c2r_runner.exec(col_ids, n_cols, dense_map_c2r<A,B>, f_avg, NULL);
			}
		}

		double data_byte = 1.0 * sizeof(A) * n_rows * n_cols;
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

