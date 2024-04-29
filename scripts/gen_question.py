from openkaito.tasks import random_eth_denver_segments, generate_question_from_eth_denver_segments

import openai
import os


def gen_question():
    root_dir = __file__.split("scripts")[0]
    print(root_dir)
    dataset_dir = root_dir + "datasets/eth_denver_dataset"
    eth_denver_dataset_dir = dataset_dir

    llm_client = openai.OpenAI(
        api_key="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",
        organization="",
        max_retries=3,
    )

    print(f"eth_denver_dataset_dir::::{eth_denver_dataset_dir}")

    # segments = random_eth_denver_segments(
    #     eth_denver_dataset_dir, num_sources=3
    # )

    # print(f'segment::::::::::::::{segments}')
    #
    # question = generate_question_from_eth_denver_segments(
    #     llm_client, segments
    # )
    #
    # print(f'question:::::::::::::::{question}')

if __name__ == '__main__':
    gen_question()
