### 运行

环境：python3，scipy, sklearn, numpy, pandas 这些常见库，加上 nltk，nltk需要额外下一个stopwords，如果没有的话按提示下就行了。

 先把解压好的 test.csv 和 train.csv 放到 data 文件夹中，再创建一个 output 文件夹，运行reproduce.py即可复现结果。预测结果会输出到 output 文件夹中。

如果想尝试其他的配置，大概有以下几个方面：

- 输入数据，改 main.py 中的 prepare，训练输入和测试输入要分别命名为 {name} 和  {name}_test
- 超参数，在util.py里，如SVM，DT的超参，训练集大小，ada_boost 迭代轮数等等。
- 弱分类器，可以改 util.py 的 models,args,names 这三个变量。
- 集成学习算法，可以新增一个py文件，写相应的 build_models 和 predict 函数即可。



改完之后，首先在相应集成学习方法的文件里跑 build_models，然后到main.py 里跑 valid，如果感觉可以就跑 submit。虽然每次需要手动写执行什么函数，但总体来说也不算太麻烦。



做过的一些实验及其结果都写在main.py的main函数里，已经都注释掉了。