from torch.utils.cpp_extension import load_inline

def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs, with_cuda=True,
                       extra_cuda_cflags=["-O3"] if opt else [], verbose=verbose, name="inline_ext")

