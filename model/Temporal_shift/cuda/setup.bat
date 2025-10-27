
conda activate pytorch

:: step1：切换到 setup.py 所在路径
cd /d D:\MyCode\ActionRecognitionAll\Shift-GCN-hb\model\Temporal_shift\cuda\

:: step2：设置环境变量
set DISTUTILS_USE_SDK=1

:: step3：执行编译
python setup.py build_ext --inplace



