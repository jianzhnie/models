# Tensorflow 将 ckpt 转成 pb 格式
使用 tf.train.saver()保存模型时会产生多个文件，会把计算图的结构和图上参数取值分成了不同的文件存储。这种方法是在TensorFlow中是最常用的保存方式。例如：下面的代码运行后，会在save目录下保存了四个文件：

```python

import tensorflow as tf

v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
v2 = tf.Variable(tf.random_normal([2, 3]), name="v2")
init_op = tf.global_variables_initializer() 
saver = tf.train.Saver() 
with tf.Session() as sess:
    sess.run(init_op)
    saver_path = saver.save(sess, "save/model.ckpt")
```
tensorflow 训练模型过程中保存的4个文件。

```python
checkpoint  model.ckpt.data-00000-of-00001  model.ckpt.index  model.ckpt.meta
```
>其中，checkpoint 是检查点文件，文件保存了一个目录下所有的模型文件列表;
>model.ckpt.meta 文件保存了TensorFlow计算图的结构，可以理解为神经网络的网络结构，
>ckpt.data 保存模型中每个变量（weights, biases, placeholders, gradients, hyper-parameters etc）的取值

很多时候，我们需要将TensorFlow的模型导出为单个文件（同时包含模型结构的定义与权重），方便在其他地方使用（如在Android中部署网络）。利用tf.train.write_graph()默认情况下只导出了网络的定义（没有权重），而利用tf.train.Saver().save()导出的文件graph_def与权重是分离的，因此需要采用别的方法。 


##  Frozen Graph

Frozen Graph 将 tensorflow 导出的模型的权重都 freeze 住，使得其都变为常量,并且模型参数和网络结构保存在同一个文件中。

### 方法一： 将计算图和模型参数绑定

TensoFlow为我们提供了convert_variables_to_constants()方法，该方法可以固化模型结构，将计算图中的变量取值以常量的形式保存，而且保存的模型可以移植到Android平台。将 ckpt 转换成 pb  格式的文件的过程可简述如下：

- 通过传入 ckpt 模型的路径得到模型的图和变量数据
- 通过 import_meta_graph 导入模型中的图
- 通过 saver.restore 从模型中恢复图中各个变量的数据
- 通过 graph_util.convert_variables_to_constants 将模型持久化
- 保存冻结的计算图和模型参数

```python
def freeze_graph(input_checkpoint,output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "InceptionV3/Logits/SpatialSqueeze"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点
```

>1、函数freeze_graph中，最重要的就是要确定“指定输出的节点名称”，这个节点名称必须是原模型中存在的节点，对于freeze操作，我们需要定义输出结点的名字。因为网络其实是比较复杂的，定义了输出结点的名字，那么freeze的时候就只把输出该结点所需要的子图都固化下来，其他无关的就舍弃掉。因为我们freeze模型的目的是接下来做预测。所以，output_node_names一般是网络模型最后一层输出的节点名称，或者说就是我们预测的目标。
>2、在保存的时候，通过convert_variables_to_constants函数来指定需要固化的节点名称，对于鄙人的代码，需要固化的节点只有一个：output_node_names。注意节点名称与张量的名称的区别，例如：“input:0”是张量的名称，而"input"表示的是节点的名称。
>3、源码中通过graph = tf.get_default_graph()获得默认的图，这个图就是由saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)恢复的图，因此必须先执行tf.train.import_meta_graph，再执行tf.get_default_graph() 。

调用方法很简单，输入ckpt模型路径，输出pb模型的路径即可：

```python
# 输入ckpt模型路径
input_checkpoint='models/model.ckpt-10000'
# 输出pb模型的路径
out_pb_path="models/pb/frozen_model.pb"
# 调用freeze_graph将ckpt转为pb
freeze_graph(input_checkpoint,out_pb_path)
```

### 方法二: 使用 tensorflow 自带的 `freeze_graph` 方法

```python
freeze_graph(input_graph=FLAGS.prototxt_file,
                        input_saver='',
                        input_binary=False,
                        input_checkpoint=FLAGS.ckpt_file,
                        output_node_names=output_node_names, # need to modify across different network
                        restore_op_name='save/restore_all',
                        filename_tensor_name='save/Const:0',
                        output_graph=FLAGS.output_pb_file,
                        clear_devices=True,
                        initializer_nodes='',
                        variable_names_blacklist='')
```

- **input_graph**：（必选）模型文件，可以是二进制的pb文件或meta文件，用input_binary来指定区分（见下面说明） 
- input_saver：（可选）Saver解析器。保存模型和权限时，Saver也可以自身序列化保存，以便在加载时应用合适的版本。主要用于版本不兼容时使用。可以为空，为空时用当前版本的Saver。 
- input_binary：（可选）配合input_graph用，为true时，input_graph为二进制，为false时，input_graph为文件。默认False 
- **input_checkpoint**：（必选）模型参数数据文件。训练时，给Saver用于保存权重、偏置等变量值。这时用于模型恢复变量值。 
- **output_node_names**：（必选）输出节点的名字，有多个时用逗号分开。用于指定输出节点，将没有在输出线上的其它节点剔除。 
- restore_op_name：（可选）从模型恢复节点的名字。默认：save/restore_all 
- filename_tensor_name：（可选）已弃用。默认：save/Const:0 
- **output_graph**：（必选）用来保存整合后的模型输出文件。 
- clear_devices：（可选），默认True。指定是否清除训练时节点指定的运算设备（如cpu、gpu、tpu。cpu是默认） 
- initializer_nodes：（可选）默认空。权限加载后，可通过此参数来指定需要初始化的节点，用逗号分隔多个节点名字。 
- variable_names_blacklist：（可先）默认空。变量黑名单，用于指定不用恢复值的变量，用逗号分隔多个变量名字。 


下面是使用 `freeze_graph` 方法导出 tensorflow 模型的主要脚本， 完整代码见 [export_model_graph.py](https://github.com/jianzhnie/models/blob/master/research/slim/export_model_graph.py)


```python
with tf.Graph().as_default() as graph:
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train',
                                        FLAGS.dataset_dir)
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=FLAGS.is_training)
    image_size = FLAGS.image_size or network_fn.default_image_size
    num_channels = 1 if FLAGS.use_grayscale else 3

    input_shape = [FLAGS.batch_size, image_size, image_size, num_channels]
    placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                shape=input_shape)
    network_fn(placeholder)

    graph_def = graph.as_graph_def()
    if FLAGS.write_text_graphdef:
        tf.io.write_graph(
            graph_def,
            os.path.dirname(FLAGS.output_prototxt_file),
            os.path.basename(FLAGS.output_prototxt_file),
            as_text=True)
    else:
        with gfile.GFile(FLAGS.output_prototxt_file, 'wb') as f:
            f.write(graph_def.SerializeToString())

freeze_graph.freeze_graph(input_graph=FLAGS.output_prototxt_file,
                        input_saver='',
                        input_binary=False,
                        input_checkpoint=FLAGS.ckpt_file,
                        output_node_names=output_node_names[FLAGS.model_name], # need to modify across different network
                        restore_op_name='save/restore_all',
                        filename_tensor_name='save/Const:0',
                        output_graph=FLAGS.output_pb_file,
                        clear_devices=True,
                        initializer_nodes='',
                        variable_names_blacklist='')
```