### 作业一、二用到的Cifar10
**直接浏览器访问这个链接就可以下载： <http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>**

压缩包**184.8MB**

**文件结构**如下：
```bash
.
├── batches.meta
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── readme.html
└── test_batch
```

**文件说明**：
data_batch_1, data_batch_2, ..., data_batch_5以及test_batch这些python序列化(object serialization)文件，都是使用了`cPickle`生成的。

可以使用以下方式进行读取：
```python
def unpickle(file):
import pickle
with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
return dict
```

每个**batch**的字典中都包含以下两个字段：
+ **data**：
    + 一个数据类型为unit8的numpy数组，shape是10000X3072；
    + 数组的每行表示一个32X32X3的彩色图像，3072中，第一个1024是红色通道，然后是绿色通道，最后1024个元素是蓝色通道，即RGB格式；
    + 图像是以行优先的格式存储，所以红色通道里的前32个值就是图像的第一行。
    + 一共是1w行（1w张图）。
+ **labels**：
    + 一个取值范围在[0,9]的有1w个元素的列表，列表中某个索引`i`对应的就是数组data中第`i`个图像的标签。

另外，**batches.meta**文件也包含一个字段对象，其内容是：
+ **label_names**：
    + 一个长度为10的列表，给出labels标号对应的语义名称。例如：label_names[0] == "airplane", label_names[1] == "automobile"等等。
