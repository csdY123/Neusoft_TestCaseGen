import os


def load_prompt_template(template_name):
    """加载带占位符的提示词模板"""
    template_path = os.path.join(os.path.dirname(__file__), 'prompts', template_name)
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read().strip()
