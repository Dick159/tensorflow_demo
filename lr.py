import tensorflow as tf
import random
import matplotlib.pyplot as plt

true_x = []

true_y = []

#生成模拟数据
for i in range(50):
    temp_x = random.uniform(i,50)
    temp_b = random.uniform(-20,20)
    temp_y = temp_x*1.52 + temp_b
    true_x.append(round(temp_x,2))
    true_y.append(temp_y)
    print(temp_b,temp_y)


sess = tf.Session()

#y = wx+b

W = tf.Variable([.3],dtype=tf.float32)

b = tf.Variable([-.3],dtype=tf.float32)

x = tf.placeholder(tf.float32)

y = W*x+b

t_y = tf.placeholder(tf.float32)

t_y_m = tf.square(y - t_y)

loss = tf.reduce_sum(t_y_m)

optimizer = tf.train.GradientDescentOptimizer(1e-5)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess.run(init)

for i in range(200):
    sess.run(train,{x:true_x,t_y:true_y})

p1,p2 = sess.run([W,b])


plt.scatter(true_x, true_y)

plt.plot(true_x,[p1*x + p2 for x in true_x],color="coral")
plt.show()