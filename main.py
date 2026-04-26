# 导入库
import requests
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 从 config 导入配置
from config import API_KEY, BASE_URL, MODEL_EP

# 请求头
headers = {
    "Authorization": API_KEY,
    "Content-Type": "application/json"
}

# ========== 步骤1：加载并处理知识库 ==========
def load_knowledge_base(file_path="knowledge.txt"):
    """加载本地知识库文件，按段落分割"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 按空行分割知识库段落
        knowledge_sections = [section.strip() for section in content.split("\n\n") if section.strip()]
        return knowledge_sections

    except Exception as e:
        print(f"读取知识库失败：{e}")
        return []

# ========== 步骤2：向量检索，找到最相关的知识库段落 ==========
def retrieve_relevant_knowledge(query, knowledge_sections, top_k=2):
    """用TF-IDF实现轻量级向量检索，返回最相关的top_k个段落"""
    # 初始化TF-IDF向量化器
    vectorizer = TfidfVectorizer()
    # 把用户问题和知识库段落一起向量化
    corpus = [query] + knowledge_sections
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # 计算用户问题和每个知识库段落的相似度
    query_vector = tfidf_matrix[0]
    knowledge_vectors = tfidf_matrix[1:]
    similarity_scores = np.dot(knowledge_vectors, query_vector.T).toarray().flatten()
    
    # 取相似度最高的top_k个段落
    top_indices = similarity_scores.argsort()[-top_k:][::-1]
    relevant_sections = [knowledge_sections[i] for i in top_indices]
    
    return "\n\n".join(relevant_sections)

# ========== 步骤3：调用大模型，结合知识库生成回答 ==========
def generate_answer(query, context):
    prompt = f"""
    你是一名英雄联盟电竞知识库助手，只能根据提供的知识库内容回答用户问题。
    如果知识库中没有相关信息，请直接回答"抱歉，我还不知道这个信息"，不要编造内容。

    知识库内容：
    {context}

    用户问题：{query}
    """

    data = {
        "model": MODEL_EP,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(BASE_URL, json=data, headers=headers)
    result = response.json()

    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    else:
        return "回答生成失败，请检查API配置"

# ========== 主程序入口 ==========
def main():
    print("=" * 40)
    print("电竞知识库问答助手 v1.0")
    print("=" * 40)

    # 加载知识库
    print("正在加载知识库...")
    knowledge_sections = load_knowledge_base()
    print(f"知识库加载完成，共 {len(knowledge_sections)} 条信息\n")

    # 问答循环
    while True:
        query = input("请输入你的问题（输入exit退出）：")
        if query.lower() == "exit":
            print("程序退出")
            break
        
        print("正在检索知识库并生成回答...\n")
        # 检索相关知识
        relevant_context = retrieve_relevant_knowledge(query, knowledge_sections)
        # 生成回答
        answer = generate_answer(query, relevant_context)
        
        print("=" * 50)
        print(f"问题：{query}")
        print(f"回答：{answer}")
        print("=" * 50 + "\n")

if __name__ == "__main__":
    main()