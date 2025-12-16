import requests

# Embedding
response = requests.post('http://localhost:11434/api/embed', json={
    'model': 'bge-large',
    'input': '这是一段测试文本'
})
embedding = response.json()['embeddings'][0]
print(f"向量维度: {len(embedding)}")