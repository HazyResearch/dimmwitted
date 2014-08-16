#include "common.h"
#include "engine/scheduler.h"
#include "engine/scheduler_strawman.h"
#include "engine/scheduler_hogwild.h"
#include "engine/scheduler_percore.h"
#include "engine/scheduler_pernode.h"

#include <xmmintrin.h>
#include <immintrin.h>
#include <avxintrin.h>

struct Example {
	long i_row;
	long i_col;
	double rating;
};

struct Chunk {
	long n_row_start;
	long n_col_start;
	long n_row_end;
	long n_col_end;
	Example * examples;
};

struct JellyfishData{
  long n_examples;
  long n_row;
  long n_col;
  long n_chunks;
  Chunk * chunks;
};

struct JellyfishModel{
	long n_row;
	long n_col;
	long n_rank;
	double ** L;	// n_row * n_rank
	double ** R_t;	// n_col * n_rank
};

void jellyfish_map (long i_task, const JellyfishData * const rddata, JellyfishModel * const wrdata){


}

void jellyfish_comm (JellyfishModel * const a, JellyfishModel ** const b, int nreplicas){


}

void jellyfish_finalize (JellyfishModel * const a, JellyfishModel ** const b, int nreplicas){
  
}

void jellyfish_model_allocator (JellyfishModel ** const a, const JellyfishModel * const b){

}


void jellyfish_do(){

	long nrow = 10000;
	long ncol = 1000;
	long nexps = 10000;
	Example * examples = new Example[nexps];
	for(int i=0;i<nexps;i++){
		examples[i].i_row = rand() % nrow;
		examples[i].i_col = rand()
	}





}












