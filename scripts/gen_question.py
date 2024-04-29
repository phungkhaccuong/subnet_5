from dotenv import load_dotenv

from neurons.validator import Validator
from openkaito.tasks import random_eth_denver_segments, generate_question_from_eth_denver_segments

import openai
import os


def gen_question():
    root_dir = __file__.split("scripts")[0]
    dataset_dir = root_dir + "datasets/eth_denver_dataset"
    eth_denver_dataset_dir = dataset_dir


    load_dotenv()

    # for ranking results evaluation
    llm_client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        organization=os.getenv("OPENAI_ORGANIZATION"),
        max_retries=3,
    )

    segments = random_eth_denver_segments(
        eth_denver_dataset_dir, num_sources=3
    )

    segments = [{'episode_id': 'NMwkUIWRr5U',
     'episode_title': "#ETHDenver24-Neptune Stage #BUIDL-03/01 1115-Rob Solomon - DIMO-You'll Buy Your Car Onchain in 2030 2024-03-01 18:15",
     'episode_url': 'https://www.youtube.com/watch?v=NMwkUIWRr5U', 'created_at': '2024-03-01T00:00:00.000Z',
     'company_name': 'DIMO', 'segment_start_time': 2963.661, 'segment_end_time': 3054.227,
     'text': "So it's been almost a year now. So version 2 was really early. There was no cross-chain connection. So the whole idea was that all the NFT on mainnet or on other chains had a TBA account address. Then we built version 3, where now this is the current state of things. If you have an asset on mainnet, you can actually execute a transaction on Polygon and Optimism. And the thing you cannot do right now is have a Polygon NFT executed mainnet transaction, so this is not possible at the moment. And so we're actually, and we have announced earlier this week, but we're forming a partnership with Layer Zero, and they've been an incredible partner in opening up, you'll see the next slide, it's going to be mind-blowing, the amount of interconnectivity that's just going to mind-blow every NFT that's out there. So now we're in a reality where we have token-bound version 3 with Layer Zero version 2, where now you have a noun on Ethereum, and you can effectively transact not only to an L2 but L2 to back to the main net or L2 to L2 you can see how crazy those arrows are and it's about to get really really fun with this new so this thing's gonna be pushed v3 plus layer 0 version 2 very shortly I think in the next week or so And so anyway, we think that TBAs are a fundamental EVM paradigm. It's here to stay.",
     'speaker': 'Benny', 'segment_id': 28, 'doc_id': 'NMwkUIWRr5U.28'}]

    print(f'segments::::::::::::::{segments}')

    question = generate_question_from_eth_denver_segments(
        llm_client, segments
    )

    print(f'question:::::::::::::::{question}')

if __name__ == '__main__':
    gen_question()
