# 主体架构
> 参数配置文件
+ config.py
> 模型两个模块
- model.py
> 数据加载器
- MyDataSet.py
> 训练
- train.py 训练主体文件
> 测试方法
- test.py
> 数值图像化方法库
- Visualization.py
> 服务接口搭建
- app.py

# 目录结构
> 根目录
- 存储当前目录的代码文件
> workspace
- 存储结果文件以及模型参数文件

# 结果
方法|Time(ms)|Acc(%)
--|:--:|--:
Net_Greedy|0|99
Net_BeamSearch|0|99
Net_BeamSearch_Code128|0|99