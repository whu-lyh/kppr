clear
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# cd kppr/models/extensions
# sh ./install.sh
# cd ../../..
pip install -e .
cd kppr
python train.py --config config/config_PointTransformer.yaml