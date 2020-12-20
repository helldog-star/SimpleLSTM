import tensorflow as tf

batch_size = 4
sequence_length = 5
input_size = 30
output_size = 20

# LSTM's input : batch_size, sequence_size, input_size
# LSTM's output 1 : [batch_size, sequence_size, output_size]
#        output 2 : [batch_size, output_size]
class CustomLSTM(tf.keras.layers.Layer):
    """
    # LSTM's input : batch_size, sequence_size, input_size
    # LSTM's output 1 : [batch_size, sequence_size, output_size]
    #        output 2 : [batch_size, output_size]
    """
    # 定义构造函数，有一个output_size参数，构造函数中调用父类构造方法
    def __init__(self, output_size, return_sequences = False):
        super(CustomLSTM, self).__init__()
        self.output_size = output_size
        self.return_sequences = return_sequences
   
    # 定义build函数，里面用来设置model参数。build中调用父类build方法，有一个参数input_shape
    def build(self, input_shape):
        super(CustomLSTM, self).build(input_shape)
        input_size = int(input_shape[-1])
        self.wf = self.add_weight('wf', shape=(input_size, self.output_size))
        self.wi = self.add_weight('wi', shape=(input_size, self.output_size))
        self.wo = self.add_weight('wo', shape=(input_size, self.output_size))
        self.wc = self.add_weight('wc', shape=(input_size, self.output_size))
    
        self.uf = self.add_weight('uf', shape=(self.output_size, self.output_size))
        self.ui = self.add_weight('ui', shape=(self.output_size, self.output_size))
        self.uo = self.add_weight('uo', shape=(self.output_size, self.output_size))
        self.uc = self.add_weight('uc', shape=(self.output_size, self.output_size))
        
        self.bf = self.add_weight('bf', shape=(1, self.output_size))
        self.bi = self.add_weight('bi', shape=(1, self.output_size))
        self.bo = self.add_weight('bo', shape=(1, self.output_size))
        self.bc = self.add_weight('bc', shape=(1, self.output_size))
        
    def call(self, x):
        sequence_outputs = []
        for i in range(sequence_length):
            if i == 0:
                xt = x[:, 0, :]
                ft = tf.sigmoid(tf.matmul(xt, self.wf) + self.bf)
                it = tf.sigmoid(tf.matmul(xt, self.wi) + self.bi)
                ot = tf.sigmoid(tf.matmul(xt, self.wo) + self.bo)
                cht = tf.tanh(tf.matmul(xt, self.wc) + self.bc)
                ct = it * cht
                ht = ot * tf.tanh(ct)
            else:
                xt = x[:, i, :]
                ft = tf.sigmoid(tf.matmul(xt, self.wf) + tf.matmul(ht, self.uf) + self.bf)
                it = tf.sigmoid(tf.matmul(xt, self.wi) + tf.matmul(ht, self.ui) + self.bi)
                ot = tf.sigmoid(tf.matmul(xt, self.wo) + tf.matmul(ht, self.uo) + self.bo)
                cht = tf.tanh(tf.matmul(xt, self.wc) + tf.matmul(ht, self.uc) + self.bc)
                ct = ft * ct + it * cht
                ht = ot * tf.tanh(ct)
            sequence_outputs.append(ht)
            
        sequence_outputs = tf.stack(sequence_outputs)# stack 张量拼接函数
        sequence_outputs = tf.transpose(sequence_outputs, (1, 0, 2))# 转置函数
        if self.return_sequences:
            return sequence_outputs
        # 默认返回最后一维
        return sequence_outputs[:, -1, :]

# 测试代码
# x = tf.random.uniform((batch_size, sequence_length, input_size))
# lstm = CustomLSTM(output_size = output_size)
# lstm(x)

# 设置成二分类的任务
model = tf.keras.Sequential([
    CustomLSTM(output_size = 32),
    tf.keras.layers.Dense(2, activation = 'softmax')
])
# 设置一个损失函数
model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam()
)

x_data = tf.random.uniform((batch_size*1000, sequence_length, input_size))
y_data = tf.random.uniform((batch_size*1000,) , maxval=2, dtype=tf.int32)

model.fit(x_data, y_data, batch_size=4)

#model.fit(x_data, y_data, batch_size=4)

#model.fit(x_data, y_data, batch_size=4)

