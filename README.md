# LLM Router
This project builds a small ML system that decides which LLM should answer a user query. It learns from past model comparisons to predict whether a strong model or a cheaper model is sufficient.

---
### How it works
1. **Dataset preparation**  
   The project uses the Chatbot Arena dataset, which contains prompts and the model that produced the better answer. From this dataset we extract the user query, compared models, and the winning model.

2. **Query embeddings**  
   Each query is converted into a numeric vector using the OpenAI embedding model.

3. **Router model training**  
   A matrix-factorization model is trained to predict the probability that a strong model will perform better for a given query.

4. **Routing decision**  
   When a new prompt arrives:
   - the query is embedded
   - the router predicts the probability that a strong model is needed
   - if the probability is high: use a strong LLM  
   - otherwise: use a cheaper model