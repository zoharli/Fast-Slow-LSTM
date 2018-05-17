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

    def __init__(self, config, input_size,num_units, f_bias=1.0, use_zoneout=False,
                 zoneout_keep_h = 0.9, zoneout_keep_c = 0.5, is_training = False,use_i_mem=False,use_h_mem=False):
        """Initialize the Layer Norm LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (default 1.0).
          use_recurrent_dropout: float, Whether to use Recurrent Dropout (default False)
          dropout_keep_prob: float, dropout keep probability (default 0.90)
        """
        self.config=config
        self.input_size=input_size
        self.num_units = num_units
        self.f_bias = f_bias

        self.use_zoneout  = use_zoneout
        self.zoneout_keep_h = zoneout_keep_h
        self.zoneout_keep_c = zoneout_keep_c

        self.is_training = is_training
        self.use_i_mem=use_i_mem
        self.use_h_mem=use_h_mem

        self.memcnt=0
        self.tau=1.

        mode=config.mode
        if mode=='train':
            batch_size=config.batch_size
        elif mode=='val':
            batch_size=config.val_batch_size
        elif mode=='test':
            batch_size=config.test_batch_size
        self.batch_size=batch_size 
        if self.i_mem:
            self.i_auxcell = LN_LSTMCell(config.head_size+1,use_zoneout=True,is_training=is_training)
        if self.h_mem:
            self.h_auxcell = LN_LSTMCell(config.head_size+1,use_zoneout=True,is_training=is_training)

        self.i_mem=None
        self.h_mem=None


    def _reset_mem(self):
        self.memcnt=0
        if self.use_i_mem:
            self.i_mem=tf.zeros(self.batch_size,self.mem_cap,self.input_size)
            self.i_last_use=tf.ones(self.batch_size,self.mem_cap)*-9999999.)
        if self.use_h_mem:
            self.h_mem=tf.zeros(self.batch_size,self.mem_cap,self.num_units)
            self.h_last_use=tf.ones(self.batch_size,self.mem_cap)*-9999999.)

    def set_tau(self,num):
        self.tau=num

    def write(self,entry,index,mem,scope=None):
        with tf.name_scope(scope):
            ones=tf.expand_dims(index,axis=2)
            mem=entry.unsqueeze(1)*ones+mem*(1.-ones)
            return mem

    def read(self,x,h,last_use,ux_state,cell,mem,scope=None):
        with tf.variable_scope(scope):
            read_key,new_aux_state=cell(tf.concat(axis=1,values=[x,h]),aux_state)
            assert len(read_head.shape)==2
            read_key,time=tf.split(read_key,[self.head_size,1],axis=1)
            read_key=tf.expand_dims(read_key,axis=1)
            key=tf.contrib.layers.legacy_fully_connected(tf.stop_gradient(mem),config.head_size//2)
            key=tf.contrib.layers.legacy_fully_connected(key,config.head_size,tf.sigmoid)
            read_head=1/(1e-8+tf.sqrt(tf.reduce_sum((read_key-key)**2,axis=2)+self.config.time_fac*(time-last_use)**2))
            index=gumbel_softmax(read_head,self.tau)
            entry=tf.reduce_sum(tf.expand_dims(index,axis=2)*mem,axis=1)
        return entry,index,new_aux_state

    def __call__(self, x, state, aux_state=(None,None),scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            x0=x
            h, c = state
            h_detach=tf.stop_grad(h)
            x_detach=tf.stop_grad(x)
            i_aux_state=aux_state[0]
            h_aux_state=aux_state[1]
            
            h_size = self.num_units
            x_size = x.get_shape().as_list()[1]
            
            w_init = aux.orthogonal_initializer(1.0)
            h_init = aux.orthogonal_initializer(1.0)
            b_init = tf.constant_initializer(0.0)

            W_xh = tf.get_variable('W_xh',
                                   [x_size if self.i_mem else 2*x_size, 4 * h_size], initializer=w_init, dtype=tf.float32)
            W_hh = tf.get_variable('W_hh',
                                   [h_size if self.h_mem else 2*h_size, 4 * h_size], initializer=h_init, dtype=tf.float32)
            bias = tf.get_variable('bias', [4 * h_size], initializer=b_init, dtype=tf.float32)
            
            i_new_aux_state=None
            if self.use_i_mem:
                i_entry,i_index,i_new_aux_state=self.read(x_detach,h_detach,self.i_last_use,i_aux_state,i_auxcell,self.i_mem,scope='i_read')
                x=tf.concat(axis=1,values=[x,i_entry])       
                self.i_last_use-=1
                self.i_last_use-=self.i_last_use*i_index
            h_new_aux_state=None
            if self.use_h_mem:
                h_entry,h_index,h_new_aux_state=self.read(x_detach,h_detach,self.h_last_use,h_aux_state,h_auxcell,self.h_mem,scope='h_read')
                h=tf.concat(axis=1,values=[h,h_entry])
                self.h_last_use-=1
                self.h_last_use-=self.h_last_use*h_index

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
            
            if self.memcnt<self.memcap:
                h_write_index=i_write_index=tf.expand_dims(tf.concat(axis=0,values=[torch.zeros([self.memcnt]),tf.ones([1]),tf.zeros([self.memcap-1-self.memcnt])]),axis=0)
                self.memcnt+=1
            else:
                h_write_index=h_index if self.use_h_mem else None
                i_write_index=i_index if self.use_i_mem else None
            if self.use_i_mem:
                self.i_mem=self.write(x0,i_write_index,scope='i_write')
            if self.use_h_mem:
                self.h_mem=self.write(new_h,h_write_index,scope='h_write')
            
        return new_h, (new_h, new_c),(i_new_aux_state,h_new_aux_state)

    def zero_state(self, batch_size, dtype):
        h = tf.zeros([batch_size, self.num_units], dtype=dtype)
        c = tf.zeros([batch_size, self.num_units], dtype=dtype)
        aux_h = tf.zeros([batch_size, self.head_size+1], dtype=dtype)
        aux_c = tf.zeros([batch_size, self.head_size+1], dtype=dtype)
        return ((h, c),((aux_h,aux_c) if self.use_i_mem else None,(aux_h,aux_c) if slef.use_h_mem else None))
