import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
from dotenv import load_dotenv
import time

load_dotenv()

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:# return type should be str
    time.sleep(10)
    return await openai_complete_if_cache(
        "Qwen/Qwen2.5-7B-Instruct",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("API_KEY"),
        base_url="https://api.siliconflow.cn/v1",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:# return type should be np.ndarray
    time.sleep(10)
    return await openai_embedding(
        texts,
        model="BAAI/bge-m3",
        api_key=os.getenv("API_KEY"),
        base_url="https://api.siliconflow.cn/v1"
    )


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)


asyncio.run(test_funcs())


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=embedding_func
    ),
)


with open("/home/qy/keyan/nosft/data/raw_data/2022-06-08-enriched.txt") as f:
    rag.insert(f.read())

# Perform naive search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
)

# Perform local search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
)

# Perform global search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
)

# Perform hybrid search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
)
