from openkaito.tasks import random_eth_denver_segments, generate_question_from_eth_denver_segments

import openai
import os


def gen_question():
    root_dir = __file__.split("scripts")[0]
    dataset_dir = root_dir + "datasets/eth_denver_dataset"
    eth_denver_dataset_dir = dataset_dir


    llm_client = openai.OpenAI(
        api_key='sk-proj-u8ANkfogUxibhGknsui1T3BlbkFJHy9MH9wBXq5OJKGfGNX2',
        organization='sM',
        max_retries=3,
    )

    segments = random_eth_denver_segments(
        eth_denver_dataset_dir, num_sources=3
    )

    print(f'segments::::::::::::::{segments}')

    question = generate_question_from_eth_denver_segments(
        llm_client, segments
    )

    print(f'question:::::::::::::::{question}')

if __name__ == '__main__':
    gen_question()
