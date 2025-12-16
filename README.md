# api
## 方式1：直接运行
python api.py
## 方式2：使用 uvicorn（支持热重载）
uvicorn api:app --host 0.0.0.0 --port 8080 --reload

端点	方法	功能
/api/init	POST	初始化 vLLM 模型
/api/status	GET	检查服务状态
/api/features	POST	从 PRD 提取功能点
/api/test-points	POST	为功能点生成测试点
/api/test-cases	POST	为测试点生成测试用例
/api/full-pipeline	POST	完整流水线（一键生成全部）

# gradio
python app.py

cat /etc/systemd/system/snap.docker.dockerd.service.d/proxy.conf && sudo systemctl daemon-reload && sudo snap restart docker

# docker
先下载bge-large和bge-m3两个embedding模型：
docker exec ollama ollama pull bge-large && docker exec ollama ollama pull bge-m3
docker exec ollama ollama list
docker run -d --gpus all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama run qwen3:8b