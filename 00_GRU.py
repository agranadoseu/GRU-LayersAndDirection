"""
Based on eaxmples of how to execute existing implementation of GRU and GRUCell

Written by:
Alejandro Granados
School of Biomedical Engineering and Patient Sciences, King's College London, UK

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com

█←↑→↓⬤ 🡤🡥🡧🡦
"""

import numpy as np
import torch
import torch.nn as nn

input_size = 10
hidden_size = 20
seq_length = 5
batch_size = 3


"""
=====================================================================================================================
GRU ::. one layer - unidirectional
It only uses one GRUCell (█) which is shared at each time step

             y1  y2  y3  y4  y5
              ↑   ↑   ↑   ↑   ↑    
        h_0 → █ → █ → █ → █ → █ → h_t
              ↑   ↑   ↑   ↑   ↑
             x1  x2  x3  x4  x5
"""
num_layers = 1
num_directions = 1
gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False if num_directions==1 else True)

x = torch.randn(seq_length, batch_size, input_size)
h_0 = torch.randn(num_layers*num_directions, batch_size, hidden_size)
y, h_t = gru(input=x, hx=h_0)    # y[seq_length, batch_size, num_directions*hidden_size]

print('\n\n===========================================================================================================')
print('GRU ::. {} layer(s) & {}'.format(num_layers, 'unidirectional' if num_directions==1 else 'bidirectional'))
print('parameters = {}'.format(sum(p.numel() for p in gru.parameters() if p.requires_grad)))
for name, param in gru.named_parameters():
    print('\t{}[{}]'.format(name, param.size()))
print('x[{}], h_0[{}] y[{}]'.format(x.size(), h_0.size(), y.size()))
print('y[{}][{}] = \n{}'.format(y.size(), seq_length-1, y[seq_length-1]))
print('h_t[{}] = \n{}'.format(h_t.size(), h_t))


"""
GRUCell ::. one layer - unidirectional
This is equivalent to the GRU model, here we do the sequence by hand

             y1  y2  y3  y4  y5
              ↑   ↑   ↑   ↑   ↑    
        h_0 → █ → █ → █ → █ → █ → h_t
              ↑   ↑   ↑   ↑   ↑
             x1  x2  x3  x4  x5
"""
y_hat = torch.zeros((seq_length, batch_size, num_directions*hidden_size))
h_t_hat = torch.zeros((num_layers*num_directions, batch_size, hidden_size))

cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

# make sure the weights are the same for comparison purposes with GRU model above
cell.weight_ih = gru.weight_ih_l0
cell.weight_hh = gru.weight_hh_l0
cell.bias_ih = gru.bias_ih_l0
cell.bias_hh = gru.bias_hh_l0

h_t = h_0[0,:,:]     # [batch_size, hidden_size] i.e. no num_layers*num_directions as additional first dimension
for i in range(seq_length):
    h_t = cell(x[i], h_t)
    y_hat[i] = h_t

h_t_hat[0] = h_t

print('\n\nGRUCell ::. 1 layer & unidirectional')
print('parameters = {}'.format(sum(p.numel() for p in cell.parameters() if p.requires_grad)))
for name, param in cell.named_parameters():
    print('\t{}[{}]'.format(name, param.size()))
print('x[{}], h_0[{}] y_hat[{}]'.format(x.size(), h_0.size(), y_hat.size()))
print('y_hat[{}][{}] = \n{}'.format(y_hat.size(), seq_length-1, y_hat[seq_length-1]))
print('h_t_hat[{}] = \n{}'.format(h_t_hat.size(), h_t_hat))


print('\n\none layer - unidirectional')
if np.all(np.around(y.detach().numpy(), decimals=3) == np.around(y_hat.detach().numpy(), decimals=3)):
    print("[PASS] output vectors of GRU and GRUCell versions are the same")
if np.all(np.around(h_t.detach().numpy(), decimals=3) == np.around(h_t_hat.detach().numpy(), decimals=3)):
    print("[PASS] hidden state of GRU and GRUCell versions are the same")


"""
=====================================================================================================================
GRU ::. 2 layers - unidirectional
It only uses two GRUCell (█) which are shared at each time step

             y1  y2  y3  y4  y5
              ↑   ↑   ↑   ↑   ↑
        h_0 → █ → █ → █ → █ → █ → h1_t
              ↑   ↑   ↑   ↑   ↑    
        h_0 → █ → █ → █ → █ → █ → h0_t
              ↑   ↑   ↑   ↑   ↑
             x1  x2  x3  x4  x5
"""
num_layers = 2
num_directions = 1
gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False if num_directions==1 else True)

x = torch.randn(seq_length, batch_size, input_size)
h_0 = torch.randn(num_layers*num_directions, batch_size, hidden_size)
y, h_t = gru(input=x, hx=h_0)

print('\n\n===========================================================================================================')
print('GRU ::. {} layer(s) & {}'.format(num_layers, 'unidirectional' if num_directions==1 else 'bidirectional'))
print('parameters = {}'.format(sum(p.numel() for p in gru.parameters() if p.requires_grad)))
for name, param in gru.named_parameters():
    print('\t{}[{}]'.format(name, param.size()))
print('x[{}], h_0[{}]'.format(x.size(), h_0.size()))
print('y[{}][{}] = \n{}'.format(y.size(), seq_length-1, y[seq_length-1]))
print('h_t[{}] = \n{}'.format(h_t.size(), h_t))


"""
GRUCell ::. 2 layers - unidirectional
This is equivalent to the GRU model, here we do the sequence by hand

             y1  y2  y3  y4  y5     (y_l1)
              ↑   ↑   ↑   ↑   ↑
     h_0[1] → █ → █ → █ → █ → █ → h1_t
              ↑   ↑   ↑   ↑   ↑
              y1  y2  y3  y4  y5    (y_l0)
              ↑   ↑   ↑   ↑   ↑    
     h_0[0] → █ → █ → █ → █ → █ → h0_t
              ↑   ↑   ↑   ↑   ↑
             x1  x2  x3  x4  x5
"""
y_hat = torch.zeros((seq_length, batch_size, num_directions*hidden_size))
h_t_hat = torch.zeros((num_layers*num_directions, batch_size, hidden_size))

y_l0 = torch.zeros((seq_length, batch_size, hidden_size))
y_l1 = torch.zeros((seq_length, batch_size, hidden_size))

cell_l0 = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
cell_l1 = nn.GRUCell(input_size=hidden_size*num_directions, hidden_size=hidden_size)

# make sure the weights are the same for comparison purposes with GRU model above
cell_l0.weight_ih = gru.weight_ih_l0
cell_l0.weight_hh = gru.weight_hh_l0
cell_l0.bias_ih = gru.bias_ih_l0
cell_l0.bias_hh = gru.bias_hh_l0
cell_l1.weight_ih = gru.weight_ih_l1
cell_l1.weight_hh = gru.weight_hh_l1
cell_l1.bias_ih = gru.bias_ih_l1
cell_l1.bias_hh = gru.bias_hh_l1

h_t_l0 = h_0[0,:,:]     # [batch_size, hidden_size] i.e. no num_layers*num_directions as additional first dimension
# print('\n\nh_t_l0 = \n{}'.format(h_t_l0))
for i in range(seq_length):
    h_t_l0 = cell_l0(x[i], h_t_l0)
    y_l0[i] = h_t_l0

h_t_l1 = h_0[1,:,:]     # [batch_size, hidden_size] i.e. no num_layers*num_directions as additional first dimension
# print('h_t_l1 = \n{}'.format(h_t_l1))
for i in range(seq_length):
    h_t_l1 = cell_l1(y_l0[i], h_t_l1)
    y_l1[i] = h_t_l1

y_hat = y_l1            # only output of last layer
h_t_hat[0] = h_t_l0
h_t_hat[1] = h_t_l1

print('\n\nGRUCell ::. 2 layer & unidirectional')
num_params_l0 = sum(p.numel() for p in cell_l0.parameters() if p.requires_grad)
num_params_l1 = sum(p.numel() for p in cell_l1.parameters() if p.requires_grad)
print('parameters = {} (l0={} + l1={})'.format(num_params_l0+num_params_l1, num_params_l0, num_params_l1))
for name, param in cell_l0.named_parameters():
    print('\tl0: {}[{}]'.format(name, param.size()))
for name, param in cell_l1.named_parameters():
    print('\tl1: {}[{}]'.format(name, param.size()))
print('x[{}], h_0[{}]'.format(x.size(), h_0.size()))
print('y_hat[{}][{}] = \n{}'.format(y_hat.size(), seq_length-1, y_hat[seq_length-1]))
print('h_t_hat[{}] = \n{}'.format(h_t_hat.size(), h_t_hat))
# print('Tensors before concatenation .............')
# print('y_l0[{}] = \n{}'.format(seq_length-1, y_l0[seq_length-1]))
# print('h_t_l0[{}] = \n{}'.format(h_t_l0.size(), h_t_l0))
# print('y_l1[{}] = \n{}'.format(seq_length-1, y_l1[seq_length-1]))
# print('h_t_l1[{}] = \n{}'.format(h_t_l1.size(), h_t_l1))

print('\n\n2 layers - unidirectional')
if np.all(np.around(y.detach().numpy(), decimals=3) == np.around(y_hat.detach().numpy(), decimals=3)):
    print("[PASS] output vectors of GRU and GRUCell versions are the same")
if np.all(np.around(h_t.detach().numpy(), decimals=3) == np.around(h_t_hat.detach().numpy(), decimals=3)):
    print("[PASS] hidden state of GRU and GRUCell versions are the same")



"""
=====================================================================================================================
GRU ::. one layer - bidirectional
It uses two GRUCell (█) which is shared at each time step

             y1  y2  y3  y4  y5
              ↑   ↑   ↑   ↑   ↑
       hr_t ← █ ← █ ← █ ← █ ← █ ← h_0
              ↑   ↑   ↑   ↑   ↑    
       h_0  → █ → █ → █ → █ → █ → hs_t
              ↑   ↑   ↑   ↑   ↑
             x1  x2  x3  x4  x5
"""
num_layers = 1
num_directions = 2
gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False if num_directions==1 else True)

x = torch.randn(seq_length, batch_size, input_size)
h_0 = torch.randn(num_layers*num_directions, batch_size, hidden_size)
# y[seq_len, batch_size, num_directions*hidden_size] h_t[num_layers*num_directions, batch_size, hidden_size]
y, h_t = gru(input=x, hx=h_0)

print('\n\n===========================================================================================================')
print('GRU ::. {} layer(s) & {}'.format(num_layers, 'unidirectional' if num_directions==1 else 'bidirectional'))
print('parameters = {}'.format(sum(p.numel() for p in gru.parameters() if p.requires_grad)))
for name, param in gru.named_parameters():
    print('\t{}[{}]'.format(name, param.size()))
print('x[{}], h_0[{}]'.format(x.size(), h_0.size()))
print('y[{}] = \n{}'.format(seq_length-1, y[seq_length-1]))
print('h_t[{}] = \n{}'.format(h_t.size(), h_t))



"""
GRUCell ::. 1 layer - bidirectional
This is equivalent to the GRU model, here we do the sequence by hand

             y1  y2  y3  y4  y5     (y_l0_reverse)
              ↑   ↑   ↑   ↑   ↑
       hr_t ← █ ← █ ← █ ← █ ← █ → h_0[1]
              ↑   ↑   ↑   ↑   ↑
              y1  y2  y3  y4  y5    (y_l0)
              ↑   ↑   ↑   ↑   ↑    
    h_0[0]  → █ → █ → █ → █ → █ → hs_t
              ↑   ↑   ↑   ↑   ↑
             x1  x2  x3  x4  x5
"""
y_hat = torch.zeros((seq_length, batch_size, num_directions*hidden_size))
h_t_hat = torch.zeros((num_layers*num_directions, batch_size, hidden_size))

y_l0 = torch.zeros((seq_length, batch_size, hidden_size))
y_l0_reverse = torch.zeros((seq_length, batch_size, hidden_size))
cell_l0 = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
cell_l0_reverse = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

# make sure the weights are the same for comparison purposes with GRU model above
cell_l0.weight_ih = gru.weight_ih_l0
cell_l0.weight_hh = gru.weight_hh_l0
cell_l0.bias_ih = gru.bias_ih_l0
cell_l0.bias_hh = gru.bias_hh_l0
cell_l0_reverse.weight_ih = gru.weight_ih_l0_reverse
cell_l0_reverse.weight_hh = gru.weight_hh_l0_reverse
cell_l0_reverse.bias_ih = gru.bias_ih_l0_reverse
cell_l0_reverse.bias_hh = gru.bias_hh_l0_reverse

h_t_l0 = h_0[0,:,:]     # [batch_size, hidden_size] i.e. no num_layers*num_directions as additional first dimension
# print('\n\nh_t_l0 = \n{}'.format(h_t_l0))
for i in range(seq_length):
    h_t_l0 = cell_l0(x[i], h_t_l0)
    y_l0[i] = h_t_l0
# h_t[0] = h_t_l0

h_t_l0_reverse = h_0[1,:,:]     # [batch_size, hidden_size] i.e. no num_layers*num_directions as additional first dimension
# print('h_t_l1 = \n{}'.format(h_t_l1))
for i in range(seq_length-1, -1, -1):
    h_t_l0_reverse = cell_l0_reverse(x[i], h_t_l0_reverse)
    y_l0_reverse[i] = h_t_l0_reverse
# h_t[1] = h_t_l0_reverse

y_hat = torch.cat((y_l0, y_l0_reverse), 2)  # concatenation of each direction in the last dimension
h_t_hat[0] = h_t_l0
h_t_hat[1] = h_t_l0_reverse

print('\n\nGRUCell ::. 1 layer & bidirectional')
num_params_l0 = sum(p.numel() for p in cell_l0.parameters() if p.requires_grad)
num_params_l0_reverse = sum(p.numel() for p in cell_l0_reverse.parameters() if p.requires_grad)
print('parameters = {} (l0={} + l0_r={})'.format(num_params_l0+num_params_l0_reverse, num_params_l0, num_params_l0_reverse))
for name, param in cell_l0.named_parameters():
    print('\tl0: {}[{}]'.format(name, param.size()))
for name, param in cell_l0_reverse.named_parameters():
    print('\tl0_r: {}[{}]'.format(name, param.size()))
print('x[{}], h_0[{}]'.format(x.size(), h_0.size()))
print('y_hat[{}][{}] = \n{}'.format(y_hat.size(), seq_length-1, y_hat[seq_length-1]))
print('h_t_hat[{}] = \n{}'.format(h_t_hat.size(), h_t_hat))
# print('Tensors before concatenation .............')
# print('y_l0[{}] = \n{}'.format(seq_length-1, y_l0[seq_length-1]))
# print('h_t_l0[{}] = \n{}'.format(h_t_l0.size(), h_t_l0))
# print('y_l0_reverse[{}] = \n{}'.format(seq_length-1, y_l0_reverse[seq_length-1]))
# print('h_t_l0_reverse[{}] = \n{}'.format(h_t_l0_reverse.size(), h_t_l0_reverse))

print('\n\n1 layer - bidirectional')
if np.all(np.around(y.detach().numpy(), decimals=3) == np.around(y_hat.detach().numpy(), decimals=3)):
    print("[PASS] output vectors of GRU and GRUCell versions are the same")
if np.all(np.around(h_t.detach().numpy(), decimals=3) == np.around(h_t_hat.detach().numpy(), decimals=3)):
    print("[PASS] hidden state of GRU and GRUCell versions are the same")



"""
=====================================================================================================================
GRU ::. two layer - bidirectional
It uses 4 GRUCell (█) which is shared at each time step


             y1  y2  y3  y4  y5     (y_l1_reverse)
              ↑   ↑   ↑   ↑   ↑
       hr_t ← █ ← █ ← █ ← █ ← █ → h_0[3]
              ↑   ↑   ↑   ↑   ↑
              y1  y2  y3  y4  y5    (y_l1)
              ↑   ↑   ↑   ↑   ↑    
    h_0[2]  → █ → █ → █ → █ → █ → hs_t
              ↑   ↑   ↑   ↑   ↑             
             y1  y2  y3  y4  y5     (y_l0_reverse)
              ↑   ↑   ↑   ↑   ↑
       hr_t ← █ ← █ ← █ ← █ ← █ → h_0[1]
              ↑   ↑   ↑   ↑   ↑
              y1  y2  y3  y4  y5    (y_l0)
              ↑   ↑   ↑   ↑   ↑    
    h_0[0]  → █ → █ → █ → █ → █ → hs_t
              ↑   ↑   ↑   ↑   ↑
             x1  x2  x3  x4  x5
"""
num_layers = 2
num_directions = 2
gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False if num_directions==1 else True)

x = torch.randn(seq_length, batch_size, input_size)
h_0 = torch.randn(num_layers*num_directions, batch_size, hidden_size)
# y[seq_len, batch_size, num_directions*hidden_size] h_t[num_layers*num_directions, batch_size, hidden_size]
y, h_t = gru(input=x, hx=h_0)

print('\n\n===========================================================================================================')
print('GRU ::. {} layer(s) & {}'.format(num_layers, 'unidirectional' if num_directions==1 else 'bidirectional'))
print('parameters = {}'.format(sum(p.numel() for p in gru.parameters() if p.requires_grad)))
for name, param in gru.named_parameters():
    print('\t{}[{}]'.format(name, param.size()))
print('x[{}], h_0[{}]'.format(x.size(), h_0.size()))
print('y[{}] = \n{}'.format(seq_length-1, y[seq_length-1]))
print('h_t[{}] = \n{}'.format(h_t.size(), h_t))



"""
GRU ::. two layer - bidirectional
It uses 4 GRUCell (█) which is shared at each time step


             y1  y2  y3  y4  y5     (y_l1_reverse)
              ↑   ↑   ↑   ↑   ↑
       hr_t ← █ ← █ ← █ ← █ ← █ → h_0[3]
              ↑   ↑   ↑   ↑   ↑
              y1  y2  y3  y4  y5    (y_l1)
              ↑   ↑   ↑   ↑   ↑    
    h_0[2]  → █ → █ → █ → █ → █ → hs_t
              ↑   ↑   ↑   ↑   ↑             
             y1  y2  y3  y4  y5     (y_l0_reverse)
              ↑   ↑   ↑   ↑   ↑
       hr_t ← █ ← █ ← █ ← █ ← █ → h_0[1]
              ↑   ↑   ↑   ↑   ↑
              y1  y2  y3  y4  y5    (y_l0)
              ↑   ↑   ↑   ↑   ↑    
    h_0[0]  → █ → █ → █ → █ → █ → hs_t
              ↑   ↑   ↑   ↑   ↑
             x1  x2  x3  x4  x5
"""
y_hat = torch.zeros((seq_length, batch_size, num_directions*hidden_size))
h_t_hat = torch.zeros((num_layers*num_directions, batch_size, hidden_size))

y_l0 = torch.zeros((seq_length, batch_size, hidden_size))
y_l0_reverse = torch.zeros((seq_length, batch_size, hidden_size))
y_l1 = torch.zeros((seq_length, batch_size, hidden_size))
y_l1_reverse = torch.zeros((seq_length, batch_size, hidden_size))

cell_l0 = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
cell_l0_reverse = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
cell_l1 = nn.GRUCell(input_size=hidden_size*num_directions, hidden_size=hidden_size)
cell_l1_reverse = nn.GRUCell(input_size=hidden_size*num_directions, hidden_size=hidden_size)

# make sure the weights are the same for comparison purposes with GRU model above
cell_l0.weight_ih = gru.weight_ih_l0
cell_l0.weight_hh = gru.weight_hh_l0
cell_l0.bias_ih = gru.bias_ih_l0
cell_l0.bias_hh = gru.bias_hh_l0
cell_l0_reverse.weight_ih = gru.weight_ih_l0_reverse
cell_l0_reverse.weight_hh = gru.weight_hh_l0_reverse
cell_l0_reverse.bias_ih = gru.bias_ih_l0_reverse
cell_l0_reverse.bias_hh = gru.bias_hh_l0_reverse
cell_l1.weight_ih = gru.weight_ih_l1
cell_l1.weight_hh = gru.weight_hh_l1
cell_l1.bias_ih = gru.bias_ih_l1
cell_l1.bias_hh = gru.bias_hh_l1
cell_l1_reverse.weight_ih = gru.weight_ih_l1_reverse
cell_l1_reverse.weight_hh = gru.weight_hh_l1_reverse
cell_l1_reverse.bias_ih = gru.bias_ih_l1_reverse
cell_l1_reverse.bias_hh = gru.bias_hh_l1_reverse

h_t_l0 = h_0[0,:,:]     # [batch_size, hidden_size] i.e. no num_layers*num_directions as additional first dimension
# print('\n\nh_t_l0[{}] = \n{}'.format(h_t_l0.size(), h_t_l0))
for i in range(seq_length):
    h_t_l0 = cell_l0(x[i], h_t_l0)
    y_l0[i] = h_t_l0

h_t_l0_reverse = h_0[1,:,:]     # [batch_size, hidden_size] i.e. no num_layers*num_directions as additional first dimension
# print('h_t_l0_reverse[{}] = \n{}'.format(h_t_l0_reverse.size(), h_t_l0_reverse))
for i in range(seq_length-1, -1, -1):
    h_t_l0_reverse = cell_l0_reverse(x[i], h_t_l0_reverse)
    y_l0_reverse[i] = h_t_l0_reverse

y_hat_l0 = torch.cat((y_l0, y_l0_reverse), 2)

h_t_l1 = h_0[2,:,:]     # [batch_size, hidden_size] i.e. no num_layers*num_directions as additional first dimension
# print('y[{}], y[0][{}]'.format(y.size(), y[0].size()))
# print('y_l0[{}], y_l0[0][{}]'.format(y_l0.size(), y_l0[0].size()))
# print('h_t_l1[{}] = \n{}'.format(h_t_l1.size(), h_t_l1))
for i in range(seq_length):
    h_t_l1 = cell_l1(y_hat_l0[i], h_t_l1)
    y_l1[i] = h_t_l1

h_t_l1_reverse = h_0[3,:,:]     # [batch_size, hidden_size] i.e. no num_layers*num_directions as additional first dimension
# print('h_t_l1_reverse[{}] = \n{}'.format(h_t_l1_reverse.size(), h_t_l1_reverse))
for i in range(seq_length-1, -1, -1):
    h_t_l1_reverse = cell_l1_reverse(y_hat_l0[i], h_t_l1_reverse)
    y_l1_reverse[i] = h_t_l1_reverse

y_hat = torch.cat((y_l1, y_l1_reverse), 2)
h_t_hat[0] = h_t_l0
h_t_hat[1] = h_t_l0_reverse
h_t_hat[2] = h_t_l1
h_t_hat[3] = h_t_l1_reverse

print('\n\nGRUCell ::. 2 layer & bidirectional')
num_params_l0 = sum(p.numel() for p in cell_l0.parameters() if p.requires_grad)
num_params_l0_reverse = sum(p.numel() for p in cell_l0_reverse.parameters() if p.requires_grad)
num_params_l1 = sum(p.numel() for p in cell_l1.parameters() if p.requires_grad)
num_params_l1_reverse = sum(p.numel() for p in cell_l1_reverse.parameters() if p.requires_grad)
print('parameters = {} (l0={} + l0_r={} + l1={} + l1_r={})'.format(
    num_params_l0+num_params_l0_reverse+num_params_l1+num_params_l1_reverse,
    num_params_l0, num_params_l0_reverse,
    num_params_l1, num_params_l1_reverse))
for name, param in cell_l0.named_parameters():
    print('\tl0: {}[{}]'.format(name, param.size()))
for name, param in cell_l0_reverse.named_parameters():
    print('\tl0_r: {}[{}]'.format(name, param.size()))
for name, param in cell_l1.named_parameters():
    print('\tl1: {}[{}]'.format(name, param.size()))
for name, param in cell_l1_reverse.named_parameters():
    print('\tl1_r: {}[{}]'.format(name, param.size()))
print('x[{}], h_0[{}]'.format(x.size(), h_0.size()))
print('y_hat[{}][{}] = \n{}'.format(y_hat.size(), seq_length-1, y_hat[seq_length-1]))
print('h_t_hat[{}] = \n{}'.format(h_t_hat.size(), h_t_hat))
# print('Tensors before concatenation .............')
# print('y_l0[{}] = \n{}'.format(seq_length-1, y_l0[seq_length-1]))
# print('h_t_l0[{}] = \n{}'.format(h_t_l0.size(), h_t_l0))
# print('y_l0_reverse[{}] = \n{}'.format(seq_length-1, y_l0_reverse[seq_length-1]))
# print('h_t_l0_reverse[{}] = \n{}'.format(h_t_l0_reverse.size(), h_t_l0_reverse))

print('\n\n2 layer - bidirectional')
if np.all(np.around(y.detach().numpy(), decimals=3) == np.around(y_hat.detach().numpy(), decimals=3)):
    print("[PASS] output vectors of GRU and GRUCell versions are the same")
if np.all(np.around(h_t.detach().numpy(), decimals=3) == np.around(h_t_hat.detach().numpy(), decimals=3)):
    print("[PASS] hidden state of GRU and GRUCell versions are the same")