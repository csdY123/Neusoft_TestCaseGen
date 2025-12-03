import json
import re


def robust_json_parse(text):
    """
    鲁棒的JSON解析函数，尝试多种策略提取JSON内容

    策略：
    1. 提取 ```json ... ``` 代码块
    2. 提取第一个完整的 {...} 或 [...] 对象
    3. 清理文本中的常见问题（多余的逗号、注释等）
    4. 使用正则表达式定位JSON边界
    """
    if not text or not text.strip():
        raise ValueError("输入文本为空")

    # 策略1: 尝试提取 ```json ... ``` 代码块
    json_code_block_pattern = r'```json\s*([\s\S]*?)\s*```'
    matches = re.findall(json_code_block_pattern, text, re.IGNORECASE)
    if matches:
        for match in matches:
            try:
                cleaned = match.strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    # 策略2: 尝试提取 ``` ... ``` 代码块（无json标记）
    code_block_pattern = r'```\s*([\s\S]*?)\s*```'
    matches = re.findall(code_block_pattern, text)
    if matches:
        for match in matches:
            try:
                cleaned = match.strip()
                # 跳过非JSON内容
                if cleaned and (cleaned[0] in ['{', '[']):
                    return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    # 策略3: 查找第一个 { 到最后一个 } 的内容
    first_brace = text.find('{')
    last_brace = text.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = text[first_brace:last_brace + 1]

        # 尝试直接解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # 策略4: 清理常见问题后再试
            try:
                # 移除注释 (// ... 和 /* ... */)
                json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
                json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

                # 移除末尾多余的逗号
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

                # 处理单引号（转为双引号）
                # 注意：这个比较激进，可能会有问题，但可以试试
                # json_str = json_str.replace("'", '"')

                return json.loads(json_str)
            except json.JSONDecodeError:
                # 策略5: 尝试逐步缩小范围，找到最大的有效JSON
                lines = json_str.split('\n')
                for i in range(len(lines), 0, -1):
                    try:
                        partial = '\n'.join(lines[:i])
                        # 确保以}结尾
                        if partial.rstrip().endswith('}'):
                            return json.loads(partial)
                    except json.JSONDecodeError:
                        continue

    # 策略6: 查找 [ 到 ] 的数组
    first_bracket = text.find('[')
    last_bracket = text.rfind(']')

    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        json_str = text[first_bracket:last_bracket + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # 所有策略都失败，抛出异常
    raise json.JSONDecodeError(
        f"无法从文本中提取有效的JSON。文本开头: {text[:200]}...",
        text,
        0
    )
