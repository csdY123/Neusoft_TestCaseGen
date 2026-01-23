import json
import re
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置部分 =================
PORT = 12349
API_KEY = "EMPTY"
BASE_URL = f"http://localhost:{PORT}/v1"
MODEL_ID = "Qwen3-8B"
MAX_WORKERS = 10  # 并发线程数
INPUT_FILE = "nqopen.train_4k.clarify_all.jsonl"
OUTPUT_FILE_KEEP = "filtered_keep.jsonl"
OUTPUT_FILE_DISCARD = "filtered_discard.jsonl"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 核心提示词 (Data Quality Analyst) =================
DQ_SYSTEM_PROMPT = """# Role
You are an expert Data Quality Analyst for an AI Search Agent. Your goal is to aggressively filter out **unnecessary, pedantic, or logic-defying clarification** data.

# Task
Evaluate the provided JSON sample (User Query + Clarification + Answers).
Determine if the clarification is **LEGITIMATE** (Keep) or **TRIVIAL/BAD** (Discard).

# Discard Criteria (If ANY of these are true, DISCARD)

1. **The "Best Answer" Test:** - Can a single, direct answer naturally combine the information from all options? 
   - *Example:* "Venue vs. City" -> DISCARD (Just say: "At PGA National in Palm Beach Gardens").
   - *Example:* "Player vs. Number" -> DISCARD (Just say: "Barry Bonds with 73 home runs").
   - *Example:* "Year vs. Specific Date" -> DISCARD (Just say the specific date).

2. **The "Constraint Ignored" Test:**
   - Did the user's query already specify the constraint that the clarification is asking about?
   - *Example:* User asks "Who sang it **first**?" -> System asks "Original or Cover?" -> DISCARD (User already said 'first').
   - *Example:* User asks "What **year**..." -> System asks "Year or Date?" -> DISCARD (User asked for year, just give the date).

3. **Part-Whole / Inclusion:**
   - Do the answers refer to a Group vs. a Lead Member, or a Movie vs. its Director?
   - *Example:* "The Police" vs. "Sting" -> DISCARD (Sting is part of The Police; just answer "The Police").

4. **Elaboration vs. Fact:**
   - Is Option B just Option A plus some extra trivia?
   - *Example:* "1842" vs. "1842, which was his last book" -> DISCARD (Just give the detailed answer).

# Keep Criteria (Only keep if...)
- The user's query is genuinely ambiguous (e.g., same name for two different people/movies).
- The answers are **mutually exclusive** and **factually distinct**.
- A single combined answer would be confusing or contradictory.

# Output Format (JSON)
{
  "analysis": "Briefly explain why it violates or passes the criteria above.",
  "decision": "KEEP" or "DISCARD"
}
"""

DQ_USER_TEMPLATE = """
# Input Data
{json_data}
"""

# ================= 辅助函数 (已修复 BUG) =================

def clean_json_response(content):
    """
    清洗 LLM 返回的内容，移除 <think> 标签，并提取 JSON 部分。
    """
    if not content:
        return None
    
    # 1. 移除 <think>...</think> 部分 (针对 DeepSeek/Qwen 等推理模型)
    # re.DOTALL 确保 . 能匹配换行符
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    
    # 2. 尝试提取 Markdown 代码块中的 JSON
    match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        content = match.group(1)
    else:
        # 3. 如果没有代码块，尝试寻找最外层的花括号 {}
        # 这能解决模型只返回纯文本 JSON 而不带 markdown 标记的情况
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx : end_idx + 1]
    
    content = content.strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 调试用：如果解析失败，打印清洗后的内容前100个字符看看发生了什么
        # print(f"JSON Decode Error. Cleaned content snippet: {content[:100]}...") 
        return None

def call_llm_quality_check(data_record):
    """
    调用 LLM 进行质量评估
    """
    input_payload = {
        "question": data_record.get("question"),
        "clarification": data_record.get("clarification"),
        "nq_answers": data_record.get("nq_answers"),
        "answers": data_record.get("answers")
    }
    
    user_content = DQ_USER_TEMPLATE.format(
        json_data=json.dumps(input_payload, ensure_ascii=False, indent=2)
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": DQ_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            # CRITICAL UPDATE: 增加 max_tokens，因为 <think> 过程占用大量 token
            # 如果太小，JSON 还没输出就被截断了
            max_tokens=4096, 
            stream=False,
        )
        content = response.choices[0].message.content
        return clean_json_response(content)
    except Exception as e:
        print(f"API Call Error: {e}")
        return None

def process_single_line(line):
    """
    处理单行数据
    """
    try:
        record = json.loads(line.strip())
        
        if "clarification" not in record:
            return None, "NO_CLARIFICATION"

        result = call_llm_quality_check(record)
        
        if result:
            record["llm_evaluation"] = result
            # 确保 decision 存在且大写
            decision = result.get("decision", "ERROR").strip().upper()
            # 简单容错，处理模型输出 "KEEP." 这种带标点的情况
            if "KEEP" in decision:
                return record, "KEEP"
            elif "DISCARD" in decision:
                return record, "DISCARD"
            else:
                return record, "ERROR"
        else:
            return record, "ERROR"
            
    except json.JSONDecodeError:
        return None, "JSON_ERR"
    except Exception as e:
        print(f"Process Error: {e}")
        return None, "Process_ERR"

# ================= 主程序 =================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    print(f"Starting processing with {MAX_WORKERS} workers...")
    
    keep_count = 0
    discard_count = 0
    error_count = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE_KEEP, 'w', encoding='utf-8') as f_keep, \
         open(OUTPUT_FILE_DISCARD, 'w', encoding='utf-8') as f_discard:
        
        lines = f_in.readlines()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_line = {executor.submit(process_single_line, line): line for line in lines}
            
            for future in tqdm(as_completed(future_to_line), total=len(lines), desc="Filtering Data"):
                record, decision = future.result()
                
                if record is None:
                    error_count += 1
                    continue

                output_line = json.dumps(record, ensure_ascii=False) + "\n"
                
                if decision == "KEEP":
                    f_keep.write(output_line)
                    keep_count += 1
                elif decision == "DISCARD":
                    f_discard.write(output_line)
                    discard_count += 1
                else:
                    # 将解析错误的数据暂存到 discard 或单独的 error 文件
                    f_discard.write(output_line)
                    error_count += 1
                    
        f_keep.flush()
        f_discard.flush()

    print("\n" + "="*30)
    print(f"Processing Complete.")
    print(f"Total Keep    : {keep_count}")
    print(f"Total Discard : {discard_count}")
    print(f"Errors (API/Parse): {error_count}")
    print(f"Clean data saved to: {OUTPUT_FILE_KEEP}")
    print("="*30)

if __name__ == "__main__":
    main()