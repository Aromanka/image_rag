from retriever import (
    hybrid_search,
    save_retrieved_images,
    search_by_caption,
    search_by_image_embedding,
)

query = "pieplines leak gas"
results = search_by_image_embedding(query=query, top_k=3)

print(results)

save_retrieved_images(results)
