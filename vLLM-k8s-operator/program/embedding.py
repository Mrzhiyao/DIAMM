from openai import OpenAI

client = OpenAI(
    api_key="sk-*****",
    base_url="***************",
)

async def get_embedding(input_text):
# compute the embedding of the text

    embedding = client.embeddings.create(
        input=input_text,
        model="conan"
    )
    # type = embedding.data[0].embedding
    # print(embedding.data[0].embedding)

    return embedding.data[0].embedding
