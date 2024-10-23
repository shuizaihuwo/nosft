#启动控制器
python -m fastchat.serve.controller  --host 0.0.0.0  --port 21003 &

#为模型工作设置环境变量，并启动
export CUDA_VISIBLE_DEVICES=0
python -m fastchat.serve.model_worker --model-path /mnt/hgfs/shared_with_ubuntu/LLM model/Qwen2.5-1.5B --model-name Qwen2.5-1.5B --num-gpus 1 --controller-address http://0.0.0.0:21003 &

#启动openai api服务器
python -m fastchat.serve.opneai_api_server --host 0.0.0.0 --port 8200 --controller-address http://0.0.0.0:21003 