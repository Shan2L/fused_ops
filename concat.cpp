#include <torch/extension.h>
#include <iostream>

std::pair<int64_t, int64_t> get_source_tensor_info(std::vector<at::Tensor>& inputs, int64_t num_inputs, int64_t dim, int64_t index)
{
    int64_t sum=0;
    for (int64_t i=0; i<num_inputs; i++){
        
        if (sum + inputs[i].size(dim) > index){
            std::cout << "index: " << index << ", tensor_index: " << i << ", inner_index: " << index - sum << std::endl;
            return std::pair<int64_t, int64_t>{i, index-sum};
        }else { 
            sum += inputs[i].size(dim);
        }
    }


    return std::pair<int64_t, int64_t>{-1, index-sum};
}

at::Tensor  concat_cpp(std::vector<at::Tensor> inputs, int64_t dim){

    // TORCH_INTERNAL_ASSERT(weight.dim())

    int64_t num_input = inputs.size();
    std::cout <<num_input <<std::endl;
    std::vector<int64_t> sum_dim(num_input);
    int64_t ndim = inputs[0].dim();
    for (int i=0; i<num_input; i++){
        sum_dim.push_back(inputs[i].size(dim));
    }

    int dim_sum = std::accumulate(sum_dim.begin(), sum_dim.end(), 0);

    std::vector<int64_t> out_shape;

    for(int i=0; i<ndim; i++){
        if (i == dim) {
            out_shape.push_back(dim_sum);
        }else {
            out_shape.push_back(inputs[0].size(i));
        }
    }

    at::Tensor output = at::empty(out_shape);
    
    for (int i=0; i<ndim; i++){
        std::cout<<output.size(i)<<std::endl;
    }

    void* out_ptr = output.mutable_data_ptr();

    int64_t numel = output.numel();
    for (int64_t ele_index=0; ele_index < numel; ele_index++){
        std::vector<int64_t> coord;
        int64_t remain_index = ele_index;
        int64_t cur_dim_index = 0;
        for (auto iter=out_shape.begin()+1; iter!=out_shape.end(); iter++){
            int64_t dim_index = remain_index / std::accumulate(iter, out_shape.end(), 1, std::multiplies<int64_t>());
            coord.push_back(dim_index);
            remain_index = remain_index - dim_index * std::accumulate(iter, out_shape.end(), 1, std::multiplies<int64_t>());
            cur_dim_index ++;
        }
        coord.push_back(ele_index % out_shape[ndim-1]);

        std::cout << "element index: " <<ele_index << " ";
        for (auto item: coord){
            std::cout <<item << ", ";
        }
        std::cout << std::endl;
        std::pair<int64_t, int64_t> src_info = get_source_tensor_info(inputs, num_input, dim, coord[dim]);
        int64_t tensor_index = src_info.first;
        int64_t inner_tensor_index = src_info.second;

        std::vector<int64_t> src_pos = coord;
        src_pos[dim] = inner_tensor_index;
        
        int64_t src_index = 0;
        for (int i=0; i<ndim; i++){
            src_index += src_pos[i] * inputs[tensor_index].stride(i); 
        }
        float src_element = ((float*)inputs[tensor_index].const_data_ptr())[src_index];

        int64_t dst_index = 0;
        for (int i=0; i<ndim; i++){
            dst_index += coord[i] * output.stride(i);
        }
        ((float*)out_ptr)[dst_index] = src_element; 



        std::cout << src_element << std::endl;

    }
    
    return output;
    


}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &concat_cpp, "Concat forward");
}