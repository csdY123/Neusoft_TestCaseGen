import json
import os
import re
import time
from openai import OpenAI

# ================= 配置区域 =================
INPUT_FILE = "/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/chensenda/codes/Neusoft/TestCaseGen/original_data/ai4test_gui_agent_eval_data_30_pred.jsonl"
OUTPUT_FILE = INPUT_FILE.replace(".jsonl", "_evaluated_v3_retry.jsonl")

# LLM 配置
PORT = 12349
API_KEY = "EMPTY"
BASE_URL = f"http://localhost:{PORT}/v1"
MODEL_ID = "Qwen3-8B"
MAX_RETRIES = 10  # 最大重试次数

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 评估提示词 (保持不变) =================
EVAL_SYSTEM_PROMPT = """# Role
你是一名资深的移动端自动化测试专家。你的任务是基于给定的上下文（PRD文档、Feature描述、测试点 Test Point、测试用例名称），严格评估模型生成的测试步骤（Pred）与标准步骤（GT）的一致性及有效性。

# Task
请针对每一组输入数据，从以下三个维度进行评估，每个维度满分为 3 分。

# Evaluation Dimensions (Total Max: 9 Points)

### 维度 1：关键参数准确性 (Parameter Accuracy)
**关注点**：输入文本（如搜索词、地点、金额等）是否符合需求。
- **3分 (完美/兼容)**：Pred 中的关键参数与 PRD/Feature/Test_Point 指定值一致，或使用了合理的未指定值。允许忽略大小写。
- **2分 (轻微偏差)**：存在极轻微拼写错误但不影响语义。
- **1分 (严重偏差)**：参数错误但类型正确（如搜了错误的城市）。
- **0分 (完全错误)**：参数缺失或完全无关。PRD指定值未匹配直接0分。

### 维度 2：步骤逻辑与可执行性 (Step Logic & Executability)
**关注点**：动作序列是否流畅、合理，UI 交互是否正确。
- **3分 (逻辑通顺)**：步骤连贯。包含合理的 WAIT 或必要的中间跳转视为加分项。
- **2分 (冗余但可用)**：有少量无意义重复操作，不致死。
- **1分 (逻辑断层)**：缺少必要步骤或顺序错误。
- **0分 (不可执行)**：UI元素不存在或序列混乱。

### 维度 3：意图达成与业务覆盖 (Intent Fulfillment)
**关注点**：是否真正完成了 PRD 和测试点要求的核心任务。
- **3分 (完全覆盖)**：覆盖所有验证要求。若GT错误而Pred修正并完成意图，给3分。
- **2分 (核心覆盖)**：完成核心路径，遗漏次要验证。
- **1分 (路径偏离)**：未触达核心功能模块。
- **0分 (未完成)**：提前终止或偏离目标。

# Output Format
请严格以 JSON 格式返回结果，不要包含 Markdown 标记，直接返回 JSON 对象：

{
  "scores": {
    "parameter_score": 整数 (0-3),
    "logic_score": 整数 (0-3),
    "intent_score": 整数 (0-3)
  },
  "total_score": 整数 (0-9),
  "reason": "简短的中文评分理由，明确指出哪个维度扣分及原因。"
}
"""

EVAL_USER_TEMPLATE = """
【PRD 需求信息】
{prd_info}

【标准步骤 (Ground Truth)】
{gt_steps}

【模型预测步骤 (Prediction)】
{pred_steps}

请评估 Pred 的质量。输出 JSON：
"""

# ================= 核心函数 =================

def call_llm_api(prd_info, gt_steps, pred_steps, temperature=0.1):
    """基础 LLM 调用函数"""
    user_content = EVAL_USER_TEMPLATE.format(
        prd_info=json.dumps(prd_info, ensure_ascii=False, indent=2),
        gt_steps=json.dumps(gt_steps, ensure_ascii=False, indent=2),
        pred_steps=json.dumps(pred_steps, ensure_ascii=False, indent=2)
    )
    print(user_content)

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=temperature, # 使用动态传入的温度
            max_tokens=2048,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

def parse_json_response(content):
    """
    解析 JSON，返回解析结果和一个布尔值 status (True=成功, False=解析失败)
    """
    default_fail_result = {
        "scores": {"parameter_score": 0, "logic_score": 0, "intent_score": 0},
        "total_score": 0,
        "reason": "解析失败"
    }

    if not content:
        return {**default_fail_result, "reason": "LLM返回为空"}, False

    # 1. 清理 Think 标签
    if "</think>" in content:
        content = content.split("</think>")[-1]

    # 2. 提取 JSON 块
    match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        content = match.group(1)

    # 3. 寻找 {}
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    if start_idx == -1 or end_idx == -1:
        return {**default_fail_result, "reason": f"未找到JSON括号: {content[:30]}..."}, False
    
    json_str = content[start_idx : end_idx + 1]

    # 4. 尝试标准解析
    try:
        data = json.loads(json_str)
        if "scores" not in data or "total_score" not in data:
            # 结构不对，视为失败
            return {**default_fail_result, "reason": "JSON缺少必要字段"}, False
        return data, True # 解析成功
    except:
        pass # 进入正则

    # 5. 正则兜底
    try:
        p_score = re.search(r'"parameter_score"\s*:\s*(\d+)', json_str)
        l_score = re.search(r'"logic_score"\s*:\s*(\d+)', json_str)
        i_score = re.search(r'"intent_score"\s*:\s*(\d+)', json_str)
        t_score = re.search(r'"total_score"\s*:\s*(\d+)', json_str)
        reason_match = re.search(r'"reason"\s*:\s*"(.*?)"', json_str, re.DOTALL)
        if not reason_match:
            reason_match = re.search(r"'reason'\s*:\s*'(.*?)'", json_str, re.DOTALL)

        if p_score and l_score and i_score:
            p_val = int(p_score.group(1))
            l_val = int(l_score.group(1))
            i_val = int(i_score.group(1))
            t_val = int(t_score.group(1)) if t_score else (p_val + l_val + i_val)
            reason_str = reason_match.group(1) if reason_match else "正则提取Reason失败"
            
            return {
                "scores": {"parameter_score": p_val, "logic_score": l_val, "intent_score": i_val},
                "total_score": t_val,
                "reason": reason_str
            }, True # 正则抢救成功
        else:
            return {**default_fail_result, "reason": "正则未匹配到关键分数"}, False
            
    except Exception as e:
        return {**default_fail_result, "reason": f"正则兜底异常: {str(e)}"}, False

def get_reliable_evaluation(prd_info, gt_steps, pred_steps):
    """
    带有重试机制的评估主入口
    """
    last_result = None
    
    for attempt in range(MAX_RETRIES):
        # 动态调整温度： 0.1 -> 0.4 -> 0.7
        current_temp = 0.0 + (attempt * 0.3)
        if current_temp > 1.0: current_temp = 1.0
        
        # 日志提示
        if attempt > 0:
            print(f"    >> [重试 {attempt}/{MAX_RETRIES-1}] 上次解析失败，尝试升温至 temp={current_temp} ...")

        # 调用
        raw_content = call_llm_api(prd_info, gt_steps, pred_steps, temperature=current_temp)
        
        # 解析
        parsed_result, is_success = parse_json_response(raw_content)
        
        last_result = parsed_result
        
        if is_success:
            return parsed_result # 成功则直接返回
        
        # 如果失败，循环会继续，进行下一次重试

    # 如果循环结束还没成功，返回最后一次失败的结果（通常是0分）
    print(f"    !! [失败] {MAX_RETRIES} 次重试均无法解析 JSON，放弃治疗。")
    return last_result

def main():
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f_out:
            for line in f_out:
                try:
                    data = json.loads(line)
                    if "episode_id" in data:
                        processed_ids.add(data["episode_id"])
                except:
                    pass
    
    print(f"已评估 {len(processed_ids)} 条数据，开始处理剩余数据...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        
        for line_num, line in enumerate(f_in):
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            episode_id = data.get("episode_id", f"line_{line_num}")
            
            if episode_id in processed_ids:
                continue

            print(f"正在评估 Episode: {episode_id} ...")
            
            prd_info = data.get("prd_info", {})
            gt_steps = data.get("steps", [])
            pred_steps = data.get("pred", [])

            # === 使用新的可靠评估函数 ===
            eval_result = get_reliable_evaluation(prd_info, gt_steps, pred_steps)
            
            data["eval_result"] = eval_result
            
            scores = eval_result.get("scores", {})
            p = scores.get("parameter_score", "-")
            l = scores.get("logic_score", "-")
            i = scores.get("intent_score", "-")
            total = eval_result.get("total_score", 0)
            reason = eval_result.get("reason", "")[:50] + "..."

            print(f"  -> [Param:{p} | Logic:{l} | Intent:{i}] Total: {total}/9 | {reason}")

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"\n所有评估完成！结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()