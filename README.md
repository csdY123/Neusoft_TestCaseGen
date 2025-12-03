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