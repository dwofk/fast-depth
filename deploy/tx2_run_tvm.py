import tvm
import numpy as np
import argparse
import os
import time

def run_model(model_dir, input_fp, output_fp, warmup_trials, run_trials, cuda, try_randin):
    # import compiled graph
    print("=> [TVM on TX2] using model files in {}".format(model_dir))
    assert(os.path.isdir(model_dir))

    print("=> [TVM on TX2] loading model lib and ptx")
    loaded_lib = tvm.module.load(os.path.join(model_dir, "deploy_lib.o"))
    if cuda:
        dev_lib = tvm.module.load(os.path.join(model_dir, "deploy_cuda.ptx"))
        loaded_lib.import_module(dev_lib)

    print("=> [TVM on TX2] loading model graph and params")
    loaded_graph = open(os.path.join(model_dir,"deploy_graph.json")).read()
    loaded_params = bytearray(open(os.path.join(model_dir, "deploy_param.params"), "rb").read())
    
    print("=> [TVM on TX2] creating TVM runtime module")
    fcreate = tvm.get_global_func("tvm.graph_runtime.create")
    ctx = tvm.gpu(0) if cuda else tvm.cpu(0)
    gmodule = fcreate(loaded_graph, loaded_lib, ctx.device_type, ctx.device_id)
    set_input, get_output, run = gmodule["set_input"], gmodule["get_output"], gmodule["run"]

    print("=> [TVM on TX2] feeding inputs and params into TVM module")
    rgb_np = np.load(input_fp) # HWC
    x = np.zeros([1,3,224,224]) # NCHW
    x[0,:,:,:] = np.transpose(rgb_np, (2,0,1))
    set_input('0', tvm.nd.array(x.astype('float32')))
    gmodule["load_params"](loaded_params)

    print("=> [TVM on TX2] running TVM module, saving output")
    run() # not gmodule.run()
    out_shape = (1, 1, 224, 224)
    out = tvm.nd.empty(out_shape, "float32")
    get_output(0, out)
    np.save(output_fp, out.asnumpy())

    print("=> [TVM on TX2] benchmarking: {} warmup, {} run trials".format(warmup_trials, run_trials))
    # run model several times as a warmup
    for i in range(warmup_trials):
        run()
        ctx.sync()

    # profile runtime using TVM time evaluator
    ftimer = gmodule.time_evaluator("run", ctx, number=1, repeat=run_trials)
    profile_result = ftimer()
    profiled_runtime = profile_result[0]
    
    print("=> [TVM on TX2] profiled runtime (in ms): {:.5f}".format(1000*profiled_runtime))

    # try randomizing input
    if try_randin:
        randin_runtime = 0 
        for i in range(run_trials):
            x = np.random.randn(1, 3, 224, 224)
            set_input('0', tvm.nd.array(x.astype('float32')))
            randin_ftimer = gmodule.time_evaluator("run", ctx, number=1, repeat=1)
            randin_profile_result = randin_ftimer()
            randin_runtime += randin_profile_result[0]
        randomized_input_runtime = randin_runtime/run_trials
        print("=> [TVM on TX2] with randomized input on every run, profiled runtime (in ms): {:.5f}".format(1000*randomized_input_runtime))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True, 
        help='path to folder with TVM-compiled model files (required)')
    parser.add_argument('--input-fp', type=str, default='data/rgb.npy', 
        help='numpy file containing input rgb data (default: data/rgb.npy')
    parser.add_argument('--output-fp', type=str, default='data/pred.npy',
        help='numpy file to store output prediction data (default: data/pred.npy')


    parser.add_argument('--warmup', type=int, default=10, 
        help='number of inference warmup trials (default: 10)')
    parser.add_argument('--run', type=int, default=100,
        help='number of inference run trials (default: 100)')
    parser.add_argument('--cuda', type=bool, default=False,
        help='run with CUDA (default: False)')

    parser.add_argument('--randin', type=bool, default=False,
        help='profile runtime while randomizing input on every run (default: False)')

    args = parser.parse_args()
    run_model(args.model_dir, args.input_fp, args.output_fp, args.warmup, args.run, args.cuda,  try_randin=args.randin)

if __name__ == '__main__':
    main()

