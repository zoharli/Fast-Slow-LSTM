import tensorflow as tf
import aux

class LN_LSTMCell(tf.contrib.rnn.RNNCell):
    """
    Layer-Norm, with Ortho Initialization and Zoneout.
    https://arxiv.org/abs/1607.06450 - Layer Norm
    https://arxiv.org/abs/1606.01305 - Zoneout
    derived from
    https://github.com/OlavHN/bnlstm
    https://github.com/LeavesBreathe/tensorflow_with_latest_papers
    https://github.com/hardmaru/supercell
    """

    def __init__(self, num_units, f_bias=1.0, use_zoneout=False,
                 zoneout_keep_h = 0.9, zoneout_keep_c = 0.5, is_training = False):
        """Initialize the Layer Norm LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (default 1.0).
          use_recurrent_dropout: float, Whether to use Recurrent Dropout (default False)
          dropout_keep_prob: float, dropout keep probability (default 0.90)
        """
        self.num_units = num_units
        self.f_bias = f_bias

        self.use_zoneout  = use_zoneout
        self.zoneout_keep_h = zoneout_keep_h
        self.zoneout_keep_c = zoneout_keep_c

        self.is_training = is_training

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            h, c = state

            h_size = self.num_units
            x_size = x.get_shape().as_list()[1]

            w_init = aux.orthogonal_initializer(1.0)
            h_init = aux.orthogonal_initializer(1.0)
            b_init = tf.constant_initializer(0.0)

            W_xh = tf.get_variable('W_xh',
                                   [x_size, 4 * h_size], initializer=w_init, dtype=tf.float32)
            W_hh = tf.get_variable('W_hh',
                                   [h_size, 4 * h_size], initializer=h_init, dtype=tf.float32)
            bias = tf.get_variable('bias', [4 * h_size], initializer=b_init, dtype=tf.float32)

            concat = tf.concat(axis=1, values=[x, h])  # concat for speed.
            W_full = tf.concat(axis=0, values=[W_xh, W_hh])
            concat = tf.matmul(concat, W_full) + bias
            concat = aux.layer_norm_all(concat, 4, h_size, 'ln')

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(axis=1, num_or_size_splits=4, value=concat)

            new_c = c * tf.sigmoid(f + self.f_bias) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(aux.layer_norm(new_c, 'ln_c')) * tf.sigmoid(o)

            if self.use_zoneout:
                new_h, new_c = aux.zoneout(new_h, new_c, h, c, self.zoneout_keep_h,
                                           self.zoneout_keep_c, self.is_training)

        return new_h, (new_h, new_c)

    def zero_state(self, batch_size, dtype):
        h = tf.zeros([batch_size, self.num_units], dtype=dtype)
        c = tf.zeros([batch_size, self.num_units], dtype=dtype)
        return (h, c)

class LN_MALSTMCell(tf.contrib.rnn.RNNCell):
    """
    Layer-Norm, with Ortho Initialization and Zoneout.
    https://arxiv.org/abs/1607.06450 - Layer Norm
    https://arxiv.org/abs/1606.01305 - Zoneout
    derived from
    https://github.com/OlavHN/bnlstm
    https://github.com/LeavesBreathe/tensorflow_with_latest_papers
    https://github.com/hardmaru/supercell
    """

    def __init__(self, config, num_units, f_bias=1.0, use_zoneout=False,
                 zoneout_keep_h = 0.9, zoneout_keep_c = 0.5, is_training = False):
        """Initialize the Layer Norm LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (default 1.0).
          use_recurrent_dropout: float, Whether to use Recurrent Dropout (default False)
          dropout_keep_prob: float, dropout keep probability (default 0.90)
        """
        self.config=config
        self.num_units = num_units
        self.f_bias = f_bias

        self.use_zoneout  = use_zoneout
        self.zoneout_keep_h = zoneout_keep_h
        self.zoneout_keep_c = zoneout_keep_c

        self.is_training = is_training

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            h, c = state

            h_size = self.num_units
            x_size = x.get_shape().as_list()[1]

            w_init = aux.orthogonal_initializer(1.0)
            h_init = aux.orthogonal_initializer(1.0)
            b_init = tf.constant_initializer(0.0)

            W_xh = tf.get_variable('W_xh',
                                   [x_size, 4 * h_size], initializer=w_init, dtype=tf.float32)
            W_hh = tf.get_variable('W_hh',
                                   [h_size, 4 * h_size], initializer=h_init, dtype=tf.float32)
            bias = tf.get_variable('bias', [4 * h_size], initializer=b_init, dtype=tf.float32)

            key

            concat = tf.concat(axis=1, values=[x, h])  # concat for speed.
            W_full = tf.concat(axis=0, values=[W_xh, W_hh])
            concat = tf.matmul(concat, W_full) + bias
            concat = aux.layer_norm_all(concat, 4, h_size, 'ln')

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(axis=1, num_or_size_splits=4, value=concat)

            new_c = c * tf.sigmoid(f + self.f_bias) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(aux.layer_norm(new_c, 'ln_c')) * tf.sigmoid(o)

            if self.use_zoneout:
                new_h, new_c = aux.zoneout(new_h, new_c, h, c, self.zoneout_keep_h,
                                           self.zoneout_keep_c, self.is_training)

        return new_h, (new_h, new_c)

    def zero_state(self, batch_size, dtype):
        h = tf.zeros([batch_size, self.num_units], dtype=dtype)
        c = tf.zeros([batch_size, self.num_units], dtype=dtype)
        return (h, c)
class MALSTMCell(nn.Module):

    def __init__(self,options,input_size, hidden_size, use_bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.time_fac=options['time_fac']
        self.weight_ih = nn.Parameter(torch.FloatTensor(2*input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(2*hidden_size, 3 * hidden_size))
        self.bias_ih = nn.Parameter(torch.FloatTensor(1,3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.FloatTensor(1,3 * hidden_size))
        self.memcnt=0
        self.memcap=options['mem_cap']
        self.head_size=options['head_size']
        mode=options['mode']
        if mode=='train':
            batch_size=options['batch_size']
        elif mode=='val':
            batch_size=options['eval_batch_size']
        elif mode=='test':
            batch_size=options['test_batch_size']
        self.batch_size=batch_size 
        self.auxcell = GRUCell(input_size+hidden_size,2*(self.head_size+1))
        self.tau=1.
        self.i_fc=nn.Sequential(
                nn.Linear(input_size,self.head_size//2),
                nn.ReLU(),
                nn.Linear(self.head_size//2,self.head_size),
                nn.Sigmoid())  
        self.h_fc=nn.Sequential(
                nn.Linear(hidden_size,self.head_size//2),
                nn.ReLU(),
                nn.Linear(self.head_size//2,self.head_size),
                nn.Sigmoid())
        
        self.last_usage=None
        self.mem=None

        self.reset_parameters()
    
    def _reset_mem(self):
        self.memcnt=0
        self.imem=Variable(torch.zeros(self.batch_size,self.memcap,self.input_size),requires_grad=True).cuda()
        self.hmem=Variable(torch.zeros(self.batch_size,self.memcap,self.hidden_size),requires_grad=True).cuda()
        self.i_last_use=Variable(torch.ones(self.batch_size,self.memcap)*-9999999.).cuda()
        self.h_last_use=Variable(torch.ones(self.batch_size,self.memcap)*-9999999.).cuda()

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def set_tau(self,num):
        self.tau=num

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for n,p in self.named_parameters():
            if 'weight' in n:
                nn.init.orthogonal_(p)
            if 'bias' in n:
                nn.init.constant_(p.data, val=0)
        
    def forward(self, input_, h_0, aux_h_0):

        i=input_
        h=h_0.detach()
        read_head=self.auxcell(torch.cat([i,h],dim=1),aux_h_0) 
        i_read_head,h_read_head=torch.split(read_head,self.head_size+1,dim=1)
        i_head_vecs=torch.cat([self.i_fc(self.imem.detach()),self.time_fac*torch.sigmoid(self.i_last_use).detach().unsqueeze(2)],dim=2)
        h_head_vecs=torch.cat([self.h_fc(self.hmem.detach()),self.time_fac*torch.sigmoid(self.h_last_use).detach().unsqueeze(2)],dim=2)
        i_read_head=1/torch.sqrt((1e-6+(i_read_head.unsqueeze(1)-i_head_vecs)**2).sum(dim=2))
        h_read_head=1/torch.sqrt((1e-6+(h_read_head.unsqueeze(1)-h_head_vecs)**2).sum(dim=2))
        i_entry,i_read_index,h_entry,h_read_index=self.read(i_read_head,h_read_head,self.tau)
        self.i_last_use.add_(-1).add_(-self.i_last_use*i_read_index)
        self.h_last_use.add_(-1).add_(-self.h_last_use*h_read_index)
        
        new_i=torch.cat([input_,i_entry],dim=1)
        new_h0=torch.cat([h_0,h_entry],dim=1)
        wi_b = torch.addmm(self.bias_ih, new_i, self.weight_ih)
        wh_b = torch.addmm(self.bias_hh, new_h0, self.weight_hh)
        ri,zi,ni=torch.split(wi_b,self.hidden_size,dim=1)
        rh,zh,nh=torch.split(wh_b,self.hidden_size,dim=1)
        r=torch.sigmoid(ri+rh)
        z=torch.sigmoid(zi+zh)
        n=torch.tanh(ni+r*nh)
        h_1=(1-z)*n+z*h_0
        
        if self.memcnt<self.memcap:
            h_write_index=i_write_index=Variable(torch.cat([torch.zeros(self.memcnt),torch.ones(1),torch.zeros(self.memcap-1-self.memcnt)]).unsqueeze(0)).cuda()
            self.memcnt+=1
        else:
            h_write_index=h_read_index
            i_write_index=i_read_index
        self.write(input_,i_write_index,h_0,h_write_index)
        
        return h_1,read_head

    def write(self,i,i_index,h,h_index):
        i_ones=i_index.unsqueeze(2)
        h_ones=h_index.unsqueeze(2)
        self.imem=i.unsqueeze(1)*i_ones+self.imem*(1.-i_ones)
        self.hmem=h.unsqueeze(1)*h_ones+self.hmem*(1.-h_ones)

    def read(self,i_read_head,h_read_head,tau):
        i_index,_=self.gumbel_softmax(i_read_head,tau)
        h_index,_=self.gumbel_softmax(h_read_head,tau)
        i_entry=i_index.unsqueeze(2)*self.imem
        h_entry=h_index.unsqueeze(2)*self.hmem
        i_entry=i_entry.sum(dim=1)
        h_entry=h_entry.sum(dim=1)
        return i_entry,i_index,h_entry,h_index

    def gumbel_softmax(self,input, tau):
            gumbel = Variable(-torch.log(1e-20-torch.log(1e-20+torch.rand(*input.shape)))).cuda()
            y=torch.nn.functional.softmax((torch.log(1e-20+input)+gumbel)*tau,dim=1)
            ymax,pos=y.max(dim=1)
            hard_y=torch.eq(y,ymax.unsqueeze(1)).float()
            y=(hard_y-y).detach()+y
            return y,pos

    def gumbel_sigmoid(self,input, tau):
            gumbel = Variable(-torch.log(1e-20-torch.log(1e-20+torch.rand(*input.shape)))).cuda()
            y=torch.sigmoid((input+gumbel)*tau)
            #hard_y=torch.eq(y,ymax.unsqueeze(1)).float()
            #y=(hard_y-y).detach()+y
            return y

