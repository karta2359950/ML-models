Install : git clone https://github.com/Masao-Someki/Conformer.git
export PYTHONPATH="$PWD/Conformer"

compile : CUDA_LAUNCH_BLOCKING=1 python3 filename.py
若不加BLOCKING=1在本地跑的時候會有RUNTIME ERROR

Reference : https://github.com/Masao-Someki/Conformer
使用Masao-Someki在github上實作的comformer
