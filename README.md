### Context:

GitLab is struggling with information overload. Their internal documentation, spread across various wikis, Confluence pages, and Google Docs, has become difficult for employees to navigate. They've approached you to create an AI-powered chatbot that can quickly answer employees' questions based on this documentation.

## Model Choices

### Embeddings (BAAI/bge-m3):
We chose **BGE-M3** for its unique multi-functionality, covering dense, multi-vector, and sparse retrieval, which makes it adaptable for a range of query types and document structures. Its multi-lingual capabilities ensure compatibility with over 100 languages, which is advantageous for diverse user bases and supports cross-lingual searches. Additionally, BGE-M3’s support for both short and long text granularities (up to 8192 tokens) enhances retrieval effectiveness across various text formats.

### LLMs (Gemini 1.5 & LLaMA 3):
While **Gemini 1.5** provides high-quality responses, its usage is limited by a quota. To balance requests, we also employ **LLaMA 3** (on Ollama), distributing tasks across both models to avoid request limits. This hybrid approach allows us to manage costs effectively while maintaining response quality.

## Handling Tables and Images

For tables and images in PDFs, we considered multiple approaches:

- **Multimodal LLMs**: These models could process text, images, and tables in one pass, simplifying the pipeline. However, resource constraints made this impractical.
  
- **Multimodal Embeddings**: Images could be embedded and stored in a multimodal vector database, enabling direct retrieval.

- **Extract and Summarize (Our Approach)**: We opted to extract tables and images from PDFs, summarize them separately, and include them in our RAG pipeline. This approach provided a balance between resource efficiency and content comprehensiveness.

## Performance Evaluation

### Quantitative Metrics:
We will assess accuracy by comparing responses against a database of predefined queries and ground-truth answers, calculating accuracy scores. Latency will be monitored with tools like **LangSmith**, acknowledging that any delays mainly arise from model quota limitations that necessitate intermittent pauses.

### Qualitative Feedback:
Alongside traditional feedback methods, we plan to occasionally retrieve responses using alternative vector databases and varied retrieval strategies, then compare them to gather insights on response quality across different configurations.
