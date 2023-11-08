import torch

""" Positional encoding embedding. Code was taken from https://github.com/yenchenlin/nerf-pytorch and https://github.com/bmild/nerf. """

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim



def freenerf_positional_encoding(x, L, step, T, is_pos = False):
    out = [x]
    #print("first: ", x.shape)
    for j in range(L):
        out.append(torch.sin(2 ** j * x))
        out.append(torch.cos(2 ** j * x))
    #print("fis: ", out[20].shape)
    L_max = 2 * 3 * L + 3
    if len(x.shape) > 2:
        out = torch.cat(out, dim=len(x.shape)-1)
        out = out.view(-1, L_max)
    else:
        out = torch.cat(out, dim=1)
    #print("second: ", out.shape)
    
    if is_pos:
        #print("step: ", step)
        out[:,int(step/T * L_max) + 3:] = 0
    if len(x.shape) > 2:
        out = out.view(x.shape[0],x.shape[1], L_max)
    #print("third: ", out.shape)
    return out

