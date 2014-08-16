
#include "common.h"
#include "engine/scheduler.h"
#include "engine/scheduler_strawman.h"
#include "engine/scheduler_hogwild.h"
#include "engine/scheduler_percore.h"

struct Variable {
  int id;
  int nrows;
  int ncols;
  int i_buf_pos;
};

struct Weight {
  int id;
  int nrows;
  int ncols;
  bool isfixed;
  int i_buf_pos;
};

struct PositionPoint {
  int vid;
  int row;
  int col;
};

struct PositionArea {
  int vid;
  int row;
  int col;
  int n_span_rows;
  int n_span_cols;
};

struct HyperEdge {
  PositionPoint output;
  int i_buf_pos;
  int n_inputs;
  int weight_id;
  int func_id;
};

struct NNData {
  Variable * variables;
  bool * _var_isevids;
  double * _var_initvalues;

  Weight * weights;
  double * _weight_initvalues;

  HyperEdge * edges;
  PositionArea * _edge_postions;
};

struct NNModel {
  double * model;
  int n_model_elements;

  double * var_values;
  double * var_gradients;
  int n_var_elements;
};

enum CNNDirection{
  FORWARD,
  BACKWARD
};

template<CNNDirection direction>
void cnn_map (long i_task, const NNData * const rddata, NNModel * const wrdata){

  const HyperEdge & edge = rddata->edges[i_task];
  const PositionArea * input_poses = &rddata->_edge_postions[edge.i_buf_pos];
  const Weight & weight = rddata->weights[edge.weight_id];
  double * const weights = &wrdata->model[weight.i_buf_pos];
  const int & func_id = edge.func_id;

  // forward
  if(direction == FORWARD){
    if(func_id < 2000){
      double dot = 0.0;
      for(int i_input=0;i_input<edge.n_inputs;i_input++){
        const PositionArea & input_pos = input_poses[i_input];
        const Variable & var = rddata->variables[input_pos.vid];
        for(int row = input_pos.row; row < input_pos.row + input_pos.n_span_rows; row ++){
          const double * const var_values = &wrdata->var_values[var.i_buf_pos + row*var.ncols];
          const double * const weight_values = &weights[(row-input_pos.row)*weight.ncols];
          for(int col = input_pos.col; col < input_pos.col + input_pos.n_span_cols; col ++){
            dot += var_values[col] * weight_values[col - input_pos.col] ; // TODO: double check SIMD
          }
        }
      }
      const PositionPoint & output = edge.output;
      const Variable & var = rddata->variables[output.vid];
      double * const var_values = &wrdata->var_values[var.i_buf_pos];
      double * const var_gradients = &wrdata->var_gradients[var.i_buf_pos];
      var_values[output.row*var.ncols + output.col] = tanh(dot);
      var_gradients[output.row*var.ncols + output.col] = 0.0;
    }
  }

  // backward
  if(direction == BACKWARD){
    //std::cout << "~" << func_id << std::endl;
    if(func_id >= 2000){  // loss layer
      const PositionPoint & output = edge.output;
      const Variable & var = rddata->variables[output.vid];
      const double & curr_value = wrdata->var_values[var.i_buf_pos + output.row * var.ncols + output.col];
      const double & init_value = rddata->_var_initvalues[var.i_buf_pos + output.row * var.ncols + output.col];
      double & grad = wrdata->var_gradients[var.i_buf_pos + output.row * var.ncols + output.col];
      grad = 2*(curr_value-init_value);
      //std::cout << "want: " << init_value << "    curr: " << curr_value << std::endl;
      //std::cout << "      grad = " << 2*(curr_value-init_value) << std::endl;
    }
    if(func_id < 2000){ // conv layer
      double dot = 0.0;
      for(int i_input=0;i_input<edge.n_inputs;i_input++){
        const PositionArea & input_pos = input_poses[i_input];
        const Variable & var = rddata->variables[input_pos.vid];
        for(int row = input_pos.row; row < input_pos.row + input_pos.n_span_rows; row ++){
          const double * const var_values = &wrdata->var_values[var.i_buf_pos + row*var.ncols];
          const double * const weight_values = &weights[(row-input_pos.row)*weight.ncols];
          for(int col = input_pos.col; col < input_pos.col + input_pos.n_span_cols; col ++){
            dot += var_values[col] * weight_values[col - input_pos.col] ; // TODO: double check SIMD
          }
        }
      }
      //std::cout << "dot = " << dot << std::endl;
      double grad1 = 4.0 * exp(-2*dot) / (1.0 + exp(-2*dot)) / (1.0 + exp(-2*dot));
      const PositionPoint & output = edge.output;
      const Variable & var = rddata->variables[output.vid];
      double * const var_gradients = &wrdata->var_gradients[var.i_buf_pos];
      grad1 = grad1 * var_gradients[output.row*var.ncols + output.col];
      //std::cout << "lastgrad = " << var_gradients[output.row*var.ncols + output.col] << std::endl;

      for(int i_input=0;i_input<edge.n_inputs;i_input++){
        const PositionArea & input_pos = input_poses[i_input];
        const Variable & var = rddata->variables[input_pos.vid];
        for(int row = input_pos.row; row < input_pos.row + input_pos.n_span_rows; row ++){
          const double * const var_values = &wrdata->var_values[var.i_buf_pos + row*var.ncols];
          double * const var_gradients = &wrdata->var_gradients[var.i_buf_pos + row*var.ncols];
          double * const weight_values = &weights[(row-input_pos.row)*weight.ncols];
          for(int col = input_pos.col; col < input_pos.col + input_pos.n_span_cols; col ++){
            weight_values[col - input_pos.col] -= 0.1 * grad1 * var_values[col]; // TODO: double check SIMD
            //std::cout << "change" << 0.1 * grad1 * var_values[col] << std::endl;
            var_gradients[col] += grad1*weight_values[col - input_pos.col];
          }
        }
      }
    }
  }

}

void cnn_comm (NNModel * const a, NNModel ** const b, int nreplicas){

}

void cnn_finalize (NNModel * const a, NNModel ** const b, int nreplicas){
  for(int i=0;i<a->n_model_elements;i++){
    a->model[i] = 0.0;
  }

  for(int j=0;j<nreplicas;j++){
    for(int i=0;i<a->n_model_elements;i++){
      a->model[i] += b[j]->model[i]/nreplicas;
    }
  }
}

void cnn_model_allocator (NNModel ** const a, const NNModel * const b){

  *a = new NNModel();
  (*a)->model = new double[b->n_model_elements];
  (*a)->n_model_elements = b->n_model_elements;
  (*a)->n_var_elements = b->n_var_elements;
  (*a)->var_values = new double[b->n_var_elements];
  (*a)->var_gradients = new double[b->n_var_elements];

  memcpy((*a)->model, b->model, sizeof(double)*b->n_model_elements);
  memcpy((*a)->var_values, b->var_values, sizeof(double)*b->n_var_elements);
  memcpy((*a)->var_gradients, b->var_gradients, sizeof(double)*b->n_var_elements);
}


void cnn_do(){

  int nvar = 100;
  int var_nrow = 1;
  int var_ncol = 2;

  NNData data;
  NNModel model;

  data.variables = new Variable[16];
  data._var_isevids = new bool[20];
  data._var_initvalues = new double[20];

  for(int i=0;i<nvar*4*3;i++){
    data._var_initvalues[i] = 0.0;
    data._var_isevids[i] = false;
  }

  int cp = 0;
  for(int i=0;i<nvar*4;i++){
    data.variables[i].id = i;
    data.variables[i].nrows = 1;
    data.variables[i].ncols = 2;
    data.variables[i].i_buf_pos = cp;

    data._var_isevids[cp] = true;
    data._var_initvalues[cp] = i%4%2*2-1;
    cp++;

    data._var_isevids[cp] = true;
    data._var_initvalues[cp] = (i%4 <= 1 ? 0 : 1)*2-1;
    cp++;
  }
  for(int i=nvar*4;i<nvar*4*2;i++){
    data.variables[i].id = i;
    data.variables[i].nrows = 1;
    data.variables[i].ncols = 1;
    data.variables[i].i_buf_pos = cp;
    for(int r=0;r<data.variables[i].nrows;r++){
      for(int c=0;c<data.variables[i].ncols;c++){
        data._var_isevids[cp] = false;
        //data._var_isevids[cp] = true;
        //data._var_initvalues[cp] = (i%4 != 0 && i%4 != 3 ? 0 : 1) * 2 - 1;
        cp++;
      }
    }
  }
  for(int i=nvar*4*2;i<nvar*4*3;i++){
    data.variables[i].id = i;
    data.variables[i].nrows = 1;
    data.variables[i].ncols = 1;
    data.variables[i].i_buf_pos = cp;
    for(int r=0;r<data.variables[i].nrows;r++){
      for(int c=0;c<data.variables[i].ncols;c++){
        data._var_isevids[cp] = true;
        data._var_initvalues[cp] = (i%4 != 0 && i%4 != 3 ? 0 : 1) * 2 - 1;
        cp++;
      }
    }
  }
  
  //for(int i=0;i<nvar*4;i++){
  //  std::cout << "(" << data._var_initvalues[data.variables[i].i_buf_pos] << ","
  //    << data._var_initvalues[data.variables[i].i_buf_pos+1] << ") -> " 
  //    << data._var_initvalues[data.variables[i+nvar*4].i_buf_pos] << std::endl;
  //}
  
  data.weights = new Weight[3];
  data.weights[0].id = 0;
  data.weights[0].id = var_nrow;
  data.weights[0].id = var_ncol;
  data.weights[0].isfixed = false;
  data.weights[0].i_buf_pos = 0;

  data.weights[0].id = 1;
  data.weights[0].id = var_nrow;
  data.weights[0].id = var_ncol;
  data.weights[0].isfixed = false;
  data.weights[0].i_buf_pos = 0;

  data.weights[0].id = 2;
  data.weights[0].id = 1;
  data.weights[0].id = 1;
  data.weights[0].isfixed = false;
  data.weights[0].i_buf_pos = 0;

  data._weight_initvalues = new double[var_nrow*var_ncol*2+1];
  for(int r=0;r<var_nrow*var_ncol*2+1;r++){
    data._weight_initvalues[r] = 1.0;
  }

  data.edges = new HyperEdge[4*nvar*2];
  data._edge_postions = new PositionArea[4*nvar*2];
  for(int i=0;i<nvar*4;i++){
    data.edges[i].output.vid = i + 4*nvar;
    data.edges[i].output.row = 0;
    data.edges[i].output.col = 0;

    data.edges[i].i_buf_pos = i;
    data.edges[i].n_inputs = 1;
    data.edges[i].weight_id = 0;
    data.edges[i].func_id = 1000;

    data._edge_postions[i].vid = i;
    data._edge_postions[i].row = 0;
    data._edge_postions[i].col = 0;
    data._edge_postions[i].n_span_rows = var_nrow;
    data._edge_postions[i].n_span_cols = var_ncol;
  }

  for(int i=nvar*4;i<2*4*nvar;i++){
    data.edges[i].output.vid = i;
    data.edges[i].output.row = 0;
    data.edges[i].output.col = 0;
    data.edges[i].i_buf_pos = -1;
    data.edges[i].n_inputs = -1;
    data.edges[i].weight_id = -1;
    data.edges[i].func_id = 2000;
  }

  long * tasks_layer1 = new long[nvar*4];
  for(int i=0;i<nvar*4;i++){
    tasks_layer1[i] = i;
  }

  long * tasks_layer2 = new long[nvar*4];
  for(int i=nvar*4;i<2*4*nvar;i++){
    tasks_layer2[i-4*nvar] = i;
  }

  model.model = new double[var_nrow*var_ncol];
  for(int i=0;i<var_nrow*var_ncol;i++){
    model.model[i] = data._weight_initvalues[i];
  }
  model.n_model_elements = var_nrow*var_ncol;
  model.n_var_elements = nvar*4*3;
  model.var_values = new double[nvar*4*3];
  model.var_gradients = new double[nvar*4*3];
  for(int i=0;i<nvar*4*3;i++){
    model.var_values[i] = data._var_initvalues[i];
    model.var_gradients[i] = 0;
  }

  DWRun<NNData, NNModel, SCHED_PERCORE> 
    dw(&data, &model, cnn_model_allocator);

  dw.prepare();

  std::cout << "START RUNNING!" << std::endl;

  int n_epoch = 1000;
  for(int i_epoch=0;i_epoch<n_epoch;i_epoch++){

    Timer t;

    dw.exec(tasks_layer1, nvar*4, cnn_map<FORWARD>, cnn_comm, cnn_finalize);
    dw.exec(tasks_layer2, nvar*4, cnn_map<BACKWARD>, cnn_comm, cnn_finalize);
    dw.exec(tasks_layer1, nvar*4, cnn_map<BACKWARD>, cnn_comm, cnn_finalize);

    std::cout << t.elapsed() << " seconds!" << std::endl;

    std::cout << "--------------------------" << std::endl;
    double s = 0.0;
    for(int i=0;i<model.n_model_elements;i++){
      s += model.model[i];
      std::cout << "tanh(" <<model.model[0] << "x+" << model.model[1] << "y" << ")" << std::endl;
    }
    std::cout << s << std::endl;
    std::cout << "--------------------------" << std::endl;
  }

}


void cnn_do_(){

  int nvar = 100000;
  int var_nrow = 25;
  int var_ncol = 25;

  NNData data;
  NNModel model;

  data.variables = new Variable[nvar*2];
  data._var_isevids = new bool[nvar*var_nrow*var_ncol + nvar];
  data._var_initvalues = new double[nvar*var_nrow*var_ncol + nvar];
  int cp = 0;
  for(int i=0;i<nvar;i++){
    data.variables[i].id = i;
    data.variables[i].nrows = var_nrow;
    data.variables[i].ncols = var_ncol;
    data.variables[i].i_buf_pos = cp;
    for(int r=0;r<data.variables[i].nrows;r++){
      for(int c=0;c<data.variables[i].ncols;c++){
        data._var_isevids[cp] = true;
        data._var_initvalues[cp] = 1.0;
        cp++;
      }
    }
  }
  for(int i=nvar;i<2*nvar;i++){
    data.variables[i].id = i;
    data.variables[i].nrows = 1;
    data.variables[i].ncols = 1;
    data.variables[i].i_buf_pos = cp;
    for(int r=0;r<data.variables[i].nrows;r++){
      for(int c=0;c<data.variables[i].ncols;c++){
        data._var_isevids[cp] = true;
        if(i-nvar >= nvar*0.8){
          data._var_initvalues[cp] = 0.0;
        }else{
          data._var_initvalues[cp] = 1.0;
        }
        cp++;
      }
    }
  }

  data.weights = new Weight[1];
  data.weights[0].id = 0;
  data.weights[0].id = var_nrow;
  data.weights[0].id = var_ncol;
  data.weights[0].isfixed = false;
  data.weights[0].i_buf_pos = 0;

  data._weight_initvalues = new double[var_nrow*var_ncol];
  for(int r=0;r<var_nrow;r++){
    for(int c=0;c<var_ncol;c++){
      data._weight_initvalues[r*var_ncol + c] = 0.00001;
    }
  }

  data.edges = new HyperEdge[2*nvar];
  data._edge_postions = new PositionArea[nvar];
  for(int i=0;i<nvar;i++){
    data.edges[i].output.vid = i + nvar;
    data.edges[i].output.row = 0;
    data.edges[i].output.col = 0;

    data.edges[i].i_buf_pos = i;
    data.edges[i].n_inputs = 1;
    data.edges[i].weight_id = 0;
    data.edges[i].func_id = 1000;

    data._edge_postions[i].vid = i;
    data._edge_postions[i].row = 0;
    data._edge_postions[i].col = 0;
    data._edge_postions[i].n_span_rows = var_nrow;
    data._edge_postions[i].n_span_cols = var_ncol;
  }

  for(int i=nvar;i<2*nvar;i++){
    data.edges[i].output.vid = i;
    data.edges[i].output.row = 0;
    data.edges[i].output.col = 0;
    data.edges[i].i_buf_pos = -1;
    data.edges[i].n_inputs = -1;
    data.edges[i].weight_id = -1;
    data.edges[i].func_id = 2000;
  }

  long * tasks_layer1 = new long[nvar];
  for(int i=0;i<nvar;i++){
    tasks_layer1[i] = i;
  }

  long * tasks_layer2 = new long[nvar];
  for(int i=nvar;i<2*nvar;i++){
    tasks_layer2[i-nvar] = i;
  }

  model.model = new double[var_nrow*var_ncol];
  for(int i=0;i<var_nrow*var_ncol;i++){
    model.model[i] = data._weight_initvalues[i];
  }
  model.n_model_elements = var_nrow*var_ncol;
  model.n_var_elements = nvar*var_nrow*var_ncol + nvar;
  model.var_values = new double[nvar*var_nrow*var_ncol + nvar];
  model.var_gradients = new double[nvar*var_nrow*var_ncol + nvar];
  for(int i=0;i<nvar*var_nrow*var_ncol + nvar;i++){
    model.var_values[i] = data._var_initvalues[i];
    model.var_gradients[i] = 0;
  }

  DWRun<NNData, NNModel, SCHED_PERCORE> 
    dw(&data, &model, cnn_model_allocator);

  dw.prepare();

  std::cout << "START RUNNING!" << std::endl;

  int n_epoch = 1000;
  for(int i_epoch=0;i_epoch<n_epoch;i_epoch++){

    Timer t;

    dw.exec(tasks_layer1, nvar, cnn_map<FORWARD>, cnn_comm, cnn_finalize);
    dw.exec(tasks_layer2, nvar, cnn_map<BACKWARD>, cnn_comm, cnn_finalize);
    dw.exec(tasks_layer1, nvar, cnn_map<BACKWARD>, cnn_comm, cnn_finalize);

    std::cout << t.elapsed() << " seconds!" << std::endl;

    std::cout << "--------------------------" << std::endl;
    double s = 0.0;
    for(int i=0;i<model.n_model_elements;i++){
      s += model.model[i];
    }
    std::cout << s << std::endl;
    std::cout << "--------------------------" << std::endl;
  }

}










