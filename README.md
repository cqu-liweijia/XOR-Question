# XOR-Question
异或问题

异或问题是分类问题中一个比较难的问题，它催生了多层非线性感知器的出现，所以值得实现实现。
==

利用含有一层隐层的全连接神经网络就可以做到，在这个问题中最重要的就是隐层节点数目设置，按照西瓜书上的例子，发现两个节点就可以达到分类目的，但是在实际的运行中发现至少三个节点才能分开，两个节点无法达到分类目的（与训练次数无关，猜测与激活函数有关，西瓜书例子的激活函数是理想的，即负数就为0，正数就为1）。  
