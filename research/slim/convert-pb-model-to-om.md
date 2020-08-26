# 使用 ATC 工具 将 pb 模型转 om 格式

关于 ATC 工具的介绍 [ATC 工具使用指导 ](https://support.huaweicloud.com/ti-atc-A200_3000/altasatc_16_002.html)

在环境准备好后， ATC 工具的使用是比较简单的

```shell
atc --framework=3 --model=yolov3_coco.pb --output=yolov3 --soc_version=Ascend310 --input_shape="input:1,416,416,3"
```


- 1. 模型 input 节点的 data_type， input_shape 怎么设定？
>客户根据自己需要的去冻结自己的PB模型，我们atc是支持各种格式的转换的
>对于 data_type，
- 2. 一些图像的预处理操作比如 Crop、Resize、Normalize 等 是否应该加到计算图中？
>不建议放，会增加模型的复杂度。 
我们昇腾有提供DVPP硬件的预处理和AIPP 软件处理的能力
如果要使用DVPP就需要使用我们昇腾的ACL软件框架来处理
如果用AIPP就看第五个问题答复的资料
- 3. 模型的输出节点是直接选原来的网络定义的节点还是要自己定义？
    TF - PB 不限制
    PB 到 om 我们支持指定outnode 截断转换
- 4. 模型转换过程中的 insert_op_conf 应该如何设置
 
- 5. 模型的 input_shape 和 TensorFlow 导出的 pb 模型 的input_shape 是否要一一对应？
 
现在的难点在于：
 
TF 训练的脚本冻结到PB模型，这块冻结的一些细节是什么，如何冻结出来满足昇腾芯片的要求
 
建议：
Model zoom中的网络每一个均有一个对应的算法工程师
该算法工程师是对这个特定网络冻结到PB的流程最熟悉的人
这个上升建议求助一下华为这边给一个ModelZoom的接口人，然后找到对应获取网络的算法专家来进行交流
当前我们这边是没有相关流程的教程说明的，因为不同的网络情况不同

