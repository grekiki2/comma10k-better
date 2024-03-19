import torch
from IPython.display import Markdown

def lsuv_init(model, batch, iters=5, verbose=False):
    def log_acts(i, module, input, output):
        nonlocal act_means, act_stds
        out_dc = output.cpu()
        act_means[i] = out_dc.mean().item()
        act_stds[i] = out_dc.std().item()

    idx = 0
    lsuv_layers = []
    hooks = []
    def handle(layer):
        nonlocal idx, lsuv_layers, hooks
        if len(list(layer.children()))>0:
            for l in layer:
                handle(l)
        else:
            if sum(map(len, layer.parameters())) == 0:
                return
            if type(layer) in [torch.nn.Linear, torch.nn.Conv2d]:
                lsuv_layers.append((idx, layer))
            if verbose: print(idx, layer)
            hook = layer.register_forward_hook(lambda m, inpt, outpt, i=idx: log_acts(i, m, inpt, outpt))
            hooks.append(hook)
            idx += 1

    for layer in model.children():
        handle(layer)
    
    act_means = [[] for _ in range(idx)]
    act_stds = [[] for _ in range(idx)]
    for lidx, layer in lsuv_layers:
        for _ in range(iters):
            model(batch)
            if abs(act_means[lidx])<1e-3 and abs(act_stds[lidx]-1)<1e-3:
                break
            layer.weight.data /= act_stds[lidx]
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data -= act_means[lidx]
        if verbose: print("LSUV init ", lidx, layer)
        if verbose: print(act_means[lidx])
        if verbose: print(act_stds[lidx])

    for h in hooks:
        h.remove()

def activation_logs(model):
    def log_acts(i, _module, _input, output):
        nonlocal act_means, act_stds, act_hists
        out_dc = output.cpu()
        act_means[i].append(out_dc.mean().item())
        act_stds[i].append(out_dc.std().item())
        act_hists[i].append(out_dc.abs().histc(50, 0, 10).tolist())

    idx = 0
    def handle(layer):
        nonlocal idx
        if len(list(layer.children()))>0:
            for l in layer:
                handle(l)
        else:
            if sum(map(len, layer.parameters())) == 0:
                return
            layer.register_forward_hook(lambda m, inpt, outpt, i=idx: log_acts(i, m, inpt, outpt))
            idx += 1


    for layer in model.children():
        handle(layer)

    act_means = [[] for _ in range(idx)]
    act_stds = [[] for _ in range(idx)]
    act_hists = [[] for _ in range(idx)]

    return act_means, act_stds, act_hists

def _flops(x, h, w): # malo hevristike
    if x.dim()<3: return x.numel()
    if x.dim()==4: return x.numel()*h*w

def flop_param_data(model, input_shape, verbose=False, depth=2): # should probably only be called in a .ipynb
    res = '|Module|Input|Output|Num params(K)|MFLOPS|\n|--|--|--|--|--|\n'
    tot_params,tot_Mflops = 0,0
    def log_acts(i, module, input, output):
        nonlocal res,tot_params,tot_Mflops
        nparms = sum(o.numel() for o in module.parameters())
        tot_params += nparms
        *_,h,w = output.shape
        Mflops = sum(_flops(o, h, w) for o in module.parameters())/1e6
        tot_Mflops += Mflops
        if Mflops>1_000 or nparms>1_024:
            res += f'|{type(module).__name__}|{tuple(input[0].shape)}|{tuple(output.shape)}|{nparms/1000}|{Mflops:.1f}|\n'

    idx = 0
    layers = []
    hooks = []
    def handle(layer, depth_rem):
        nonlocal idx, hooks
        if len(list(layer.children()))>0 and depth_rem>0:
            for l in layer:
                handle(l, depth_rem-1)
        else:
            if sum(map(len, layer.parameters())) == 0:
                return
            if verbose: print(idx, layer)
            layers.append(layer)
            hook = layer.register_forward_hook(lambda m, inpt, outpt, i=idx: log_acts(i, m, inpt, outpt))
            hooks.append(hook)
            idx += 1

    for layer in model.children():
        handle(layer, depth)
    
    model.eval()
    model(torch.zeros(*input_shape).cuda())
    for h in hooks:
        h.remove()
    return tot_Mflops, tot_params, Markdown(res)
