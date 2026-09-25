实验目录Hattrick-main/entry

=================================

draw.py

【脚本功能】

指定同一数据集的两个模型进行绘图

【示例输入】·	

$py draw.py --dataset 1x --method1 Hattrick-f3 --epoch1 1 --method2 Hattrick --epoch2 1

【示例输出】·

如果缓存中缺失结果，抛出错误并说明，否则根据参考图绘图。坐标，标题等都要参考


=======================

select.py

【脚本功能】

指定数据集和方法，读取model文件夹下相关的内容，根据规则选择最优秀的模型以及参数

【示例输入】·	

$py select.py --dataset 1x --method Hattrick-f3 

【示例输出】

dataset  method		epoch	h_mean	h_p1	h_p10	m_mean		m_p1	m_p10	l_mean	l_p1	l_p10

1x 		Hattrick-f3 100		0.00000	0.00000	0.00000	0.00000	0.00000	0.00000	0.00000	0.00000	0.00000	

=======================

train

【脚本功能】

指定数据集，方法，超参数，调用method文件夹下的methodname_system文件进行训练。参数的传递形式应该在methodname_system中进行指定，train脚本调用对应的代码，并根据结果进行反馈，返回正确的参数传递格式以及示例。


【示例输入】


======================
generate.py


【脚本功能】

生成指定的数据集，储存到data文件夹下。数据集名称是[number'x']的形式。如果输入为1x或者错误格式抛出错误，否则新建或清除对应文件夹，并将1x数据中的流量乘上x以后放入文件夹


【示例输入】

$py generate.py --dataset 3x

【示例输出】

generate 3x finished


=====================

infer.py


【脚本功能】
 
用指定的方法，数据和epoch进行推理，如果有缓存调用缓存，否则把推理得到的结果保存到 cache 文件夹下，在随后的draw中调用，并在终端中输出路径信息

【示例输入】

$py infer.py --dataset 1x --method Hattrick-f3 --epoch 1

【示例输出1】

infer start
cache found at [cache path]

【示例输出2】

infer start
infer finished at [cache path]


=======================
list.py

【脚本功能】

用来列出指定方法所有保存好的模型以及模型相关的参数，输入包含dataset和method两个参数。从model文件夹读取信息，对dataset下method方法所有保存的epoch，打印high,medium,low的mean，p1和p10，保留五位小数

【示例输入】

$py list.py --dataset 1x --method Hattrick-f3

【示例输出】

epoch    h_mean	h_p1	h_p10	m_mean		m_p1	m_p10	l_mean	l_p1	l_p10

100		0.00000	0.00000	0.00000	0.00000	0.00000	0.00000	0.00000	0.00000	0.00000	

101		0.00000	0.00000	0.00000	0.00000	0.00000	0.00000	0.00000	0.00000	0.00000	


=======================

baseline.py


【脚本功能】

指定数据集，用基线方法进行处理，这里是groubi，结果储存在baseresult中

【示例输入】

$py baseline.py --dataset 1x 

【示例输出】

baseline1x finished



=======================
model\
	1x\
		method1\
			epoch1\


model文件夹用来存放不同方法训练得到的模型参数


======================

data\
	1x\
	2x\
	3x\
储存数据集

===========================

method\
	hattrict-f3_system.py
	hattrict_system.py

各种方法的核心代码，可以随意访问整个文件夹。比如Hattrick_system所需的normfulfill就需要cache中的baseline结果。一般不会直接访问这些文件，但是train，draw等脚本都可能访问

=======================

baseresult\
	1x\
	2x\
	3x\
=======================
cache\
	1x\
		method1\
			epoch1\
	2x\
	3x\

cache文件夹用来存放推理得到的结果

===================
pictures\
	1x\
	2x\
	3x\

picture文件夹用来存放draw.py生成的图像，名字是method1-epoch1-method2-epoch2






从头训练Hattrick

继续训练Hattrick


从头训练Hattrick-f3


继续训练Hattrick-f3




推理

$py = 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe'
$infer = 'D:\kuroresearch\Hattrick-main\test_diff_path\full_geant_2x_hattrick_f\run_full_experiment.py'
$plot = 'D:\kuroresearch\Hattrick-main\test_diff_path\full_geant_2x_hattrick_f\plot_registered_cdf.py'

& $py $infer --stage infer --dataset 1x --seeds 490 --models '[{"name":"Hattrick","epoch":49},{"name":"Hattrick-f3","epoch":4}]' --device cuda

绘图

$py = 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe'
$infer = 'D:\kuroresearch\Hattrick-main\test_diff_path\full_geant_2x_hattrick_f\run_full_experiment.py'
$plot = 'D:\kuroresearch\Hattrick-main\test_diff_path\full_geant_2x_hattrick_f\plot_registered_cdf.py'

& $py $plot --dataset 1x --method Hattrick-f3 --epoch 4 --baseline-epoch 49 --seed 490 --device cuda


2x

训练Hattrick

$py = 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe'
$base = 'D:\kuroresearch\Hattrick-main\test_diff_path\full_geant_2x_hattrick_f\run_full_experiment.py'

Test-Path -LiteralPath $py
Test-Path -LiteralPath $base

& $py $base --stage hattrick-f3 --seeds 490 --hattrick-best-epoch 22 --hattrick-f-epochs 50 --batch-size 8 --learning-rate 0.0005 --low-budget 0.03

训练Hattrick-f3

Set-Location 'D:\kuroresearch\Hattrick-main'

$py = 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe'
$script = '.\test_diff_path\full_geant_2x_hattrick_f3\run_full_experiment.py'

& $py $script `
  --stage train `
  --seed 490 `
  --epochs 50 `
  --anneal-epochs 12 `
  --top-k 5 `
  --batch-size 8 `
  --learning-rate 0.0005 `
  --low-budget 0.03 `
  --device cuda



查看模型


$py = 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe'
$script = 'D:\kuroresearch\Hattrick-main\test_diff_path\full_geant_2x_hattrick_f\list_saved_models.py'

& $py $script --dataset 2x --method Hattrick --seed 490

推理

$py = 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe'
$infer = 'D:\kuroresearch\Hattrick-main\test_diff_path\full_geant_2x_hattrick_f\run_full_experiment.py'

& $py $infer --stage infer --dataset 2x --seeds 490 --models '[{"name":"Hattrick","epoch":56},{"name":"Hattrick-f3","epoch":37}]' --device cuda


绘图

$plot = 'D:\kuroresearch\Hattrick-main\test_diff_path\full_geant_2x_hattrick_f\plot_registered_cdf.py'

& $py $plot --dataset 2x --method Hattrick-f3 --epoch 37 --baseline-epoch 56 --seed 490 --device cuda

3x

训练

$py = "D:\kuroresearch\.venv-hattrick\Scripts\python.exe"
$hattrick3x = "D:\kuroresearch\Hattrick-main\test_diff_path\full_geant_3x_hattrick\run_full_experiment.py"

& $py $hattrick3x `
  --stage train `
  --seed 490 `
  --epochs 40 `
  --top-k 5 `
  --save-every 5 `
  --batch-size 8 `
  --learning-rate 0.0005 `
  --device cuda `
  --resume
