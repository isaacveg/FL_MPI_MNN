import sys
from MNN import expr as F, numpy as mnp
import numpy as np
import time
import random

def param_compress(model, action='random', extra_info=''):
    """
    Support multiple compress functions.
    ## top_k: 
    select top_k parameters, 
    'extra_info': the compress ratio how many params remained
    return: selected params indices.

    ## random
    randomly select parameters, 
    'extra_info': the compress ratio how many params remained
    return: random seed.

    ## to_int8(NOT IMPLEMENTED)
    Naively turn parameters from float to int8. 
    'extra_info': unsaturated or saturated
    return:
    ATTENTION: use 'dequant' to turn it back to fp32 before trainning!
    """
    if action == 'random':
        # 产生种子
        rd_seed = random.randint(0, sys.maxsize)
        return rd_seed
    elif action == 'top_k':
        pass
    else:
        raise ValueError('Compress method not exist!')

    

def dequant(model, quant_method):
    """
    (Unimplemented) Dequantum parameters to fp32 for training
    """ 
    pass


def decompress_and_aggregate(global_model, client_list, method, ratio, GLOBAL_MODEL_PATH):  
    # 展平
    glb_shapes = list()
    for p in global_model.parameters:
        glb_shapes.append(p.shape)
    shapes, para_vec = params_to_vectors(global_model.parameters)
    print(glb_shapes)
    print(shapes)
    lengths = len(para_vec)
    # print(lengths, ratio, int(lengths*ratio))
    para_count = np.zeros((lengths, 2), dtype=np.float32)
    if method == "random":
        for client in client_list:
            print("client: ", client.idx, client.seed)
            # 获取客户端随机数种子，得到其发送的参数的位置
            rd_seed = client.seed
            client_params = F.load_as_list(client.local_model_path)
            # 展平参数以累加
            _, client_vec = params_to_vectors(client_params)
            random.seed(rd_seed)
            indices = random.sample(range(lengths), int(lengths*ratio))
            # 将客户端参数向量按照 indices 抽出来
            # print(type(client_params[0]),type(client_vec[0]))
            selected_vec = np.take(client_vec, indices).astype(np.float32)
            # 更新 para_count 中对应位置的值
            para_count[indices, 0] += selected_vec
            para_count[indices, 1] += 1
        print("aggregating")
        for idx, para in enumerate(para_count):
            # 仅仅聚合传输的部分
            if para[1] > 0:
                para_vec[idx] = para[0]/para[1] 
        print("loading to model")
        params = load_params_from_vectors(para_vec, shapes)
        pa_shape = list()
        for pa in params:
            pa_shape.append(pa.shape)
        print(pa_shape)
        print("Save model")
        for i in global_model.parameters:
            i.fix_as_const() 
        F.save(params, GLOBAL_MODEL_PATH)
        global_model.load_parameters(params)
        for i in global_model.parameters:
            i.fix_as_trainable()    

 

def params_to_vectors(parameters):
    """
    Turns [Var] into 1-dim vectors. 
    return: vector length, vector
    """
    shapes, params = list(), list()
    for p in parameters:
        shapes.append(p.shape)
        params.append(np.array(p.read()).reshape(-1))
    for idx, arr in enumerate(params):
        if idx == 0:
            vec = arr
            continue
        vec = np.concatenate((vec, arr))        
    return shapes, vec
    

def load_params_from_vectors(vector, shapes):
    """
    Turns vector back to model parameter list.
    """
    params = list()
    cur = 0
    for p in shapes:
        length = 1
        for x in p:
            length *= x
        temp_matrix = vector[cur:cur+length].reshape(p)
        params.append(temp_matrix)
        cur += length
    return params
    
