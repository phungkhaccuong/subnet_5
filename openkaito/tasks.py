import json
import os
from pathlib import Path
import random
import openai
from datetime import datetime, timedelta
from traceback import print_exception
from dotenv import load_dotenv

import bittensor as bt

from .protocol import SortType, StructuredSearchSynapse, SemanticSearchSynapse
from .utils.version import get_version


def random_query(input_file="queries.txt"):
    if not os.path.exists(input_file):
        bt.logging.error(f"Queries file not found at location: {input_file}")
        exit(1)
    lines = open(input_file).read().strip().splitlines()
    return random.choice(lines)


# The twitter usernames list is from a truncated snapshot of friendtech ( https://dune.com/cryptokoryo/friendtech )
# You are welcome to suggest modifications to the list, by opening Pull Request over GitHub.
def random_twitter_username(input_file="twitter_usernames.txt", num_authors: int = 2):
    if not os.path.exists(input_file):
        bt.logging.error(f"Twitter usernames file not found at location: {input_file}")
        exit(1)
    #lines = open(input_file).read().strip().splitlines()
    lines = 'HsakaTrades','Cbb0fe','0xCaptainLevi','Vombatus_eth','HerroCrypto','dingalingts','HanweChang','0xLawliette','blknoiz06','machibigbrother','CL207','0x5f_eth','CapitalGrug','0xSisyphus','Banks','Christianeth','saliencexbt','pokerbrat2019','zhusu','ManifoldTrading','pranksy','0xmj23','lBattleRhino','LomahCrypto','RookieXBT','sayinshallah','VentureCoinist','const_phoenixed','onetimebb','pokeepandaa','crypto_bitlord7','iam4x','cryptowilliamm','Arthur_0x','xcurveth','lsp8940','TeamUnibot','Anonymoux2311','iloveponzi','0xAdelina','friendtechindex','The_Bogfather','ag_dwf','mooncat2878','fewture','steveaoki','0xBreadguy','FlipGod_xyz','8892OS','Pancakesbrah','coinn_lover','OttoSuwenNFT','fundpublic777','0revenue','matthuang','DigitsDao','cryptojamie7','CryptoDonAlt','gatiencnts','0xMakesy','ColdBloodShill','saudi_biddor','SheepOfBitmex','BasePerp','SmartBiZon','frenlend','basedkarbon','icebergy_','gainzy222','bitgoten','DujunX','lior_eth','gatitayan77','hentaiavenger66','garrytan','Pentosh1','friendtechetf','semperveritas0','CryptoKaleo','KeyboardMonkey3','jrugss','cmsholdings','PaulyPaul_eth','dw8998','0xmasiwei','inversebrah','NorthRockLP','SmallCapScience','Lutherin_eth','richieweb3','zachxbt','Herr0x','__Collapse_hp','Nadeshot','CharlotteFang77','quid_defi','MaxBid24','jenfoxxuwu','pepe','loomdart','smileycapital','Tradermayne','krybharat','mouisaac','seedphrase','CozomoMedici','0xGav','Evan_ss6','cryptocevo','natealexnft','MarioNawfal','rektfoodfarmer','BitCloutCat','DeeZe','IndexIndexFT','CirrusNFT','haralabob','PelionCap','Glug69420','frankdegods','izebel_eth','punksOTC','BigDickBull69','misterblue2000','yixie10','MoonOverlord','Rawrcapital','anes427','youyou5202','CryptoCred','slangingcoins','dpats_','frengametech','GarlamWON','thisisindeed','keung','Awawat_Trades','mevcollector','Milerbtc','QuantMeta','UniswapVillain','DrgStefanescu','okx','tier10k','loc_ji','screentimes','graciehartie','MapleLeafCap','degenharambe','0xSunNFT','owen1v9','LucaNetz','Alvin0617','yuyue_chris','farokh','chiefingza','Loopifyyy','Thatjpeg_','Spvce','FEhrsam','kmoney_69','xingzhi888','StarryNightDAO','notthreadguy','stevenyuntcap','RyanNguyenHC','0xWildWizard','Anunnaki3399','nyaxeth','TruelyNoClue','wsbmod','pruggle1','DefiSquared','0xWave','0xFrankBaum','dapanji_eth','DailyLoud','TheMoonCarl','Bitcoin_Sage','aizensou55','rektmando','cryptosiem','33Misohero33','0xfoobar','gman_eth','NFTommo','cnig69','Tyler_Did_It','friendtech33','sershokunin','iluvfishnchips_','notEezzy','wangfeng_0128','NFTAura','wigger','EB7','stablekey','WSBChairman','outpxce','NoMoreLiquidity','AlgodTrading','0xAvarice','DeFiCapitalSG','FLC_FlooringLab','player1_eth','0xbeefbowl','openfriendtech','knozaki2015','adamscochran','osf_rekt','quantbike','wagieeacc','Luckytradess','EricCryptoman','CoinGurruu','valdeande','Defi_Maestro','punk2498_','nikitabier','zhansyuu','Pool2Paulie','pedrigavifrenki','elliotrades','ElfFarmer','rootslashbin','yugoviking','roxinft','Luyaoyuan1','ValerieHar2379','youdumpidump','Zeneca','ECAP100','BasedShillBH','Jampzer','IcedcoffeeEth','DextMoon','jessee40769652','RodionBezukhov','address_eth','Friend33network','KenikDreama','pristineee','pippintudor','AviFelman','IHayato','GraysonJAllen','Zata_Zu','thiccythot_','TheNFTAsian','marat2211','mrghostinvblog','cryptomanran','CryptoApprenti1','margodoge','jebus911','breathingeth','GrimaceOdysseus','IeJoth','scottmelker'
    return random.sample(lines, num_authors)


def random_datetime(start: datetime, end: datetime):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)


def random_past_datetime(start_days_ago: int = 365, end_days_ago: int = 10):

    return random_datetime(
        datetime.now() - timedelta(days=start_days_ago),
        datetime.now() - timedelta(days=end_days_ago),
    )


def generate_author_index_task(
    size: int = 5,
    num_authors: int = 2,
):
    author_usernames = random_twitter_username(num_authors=num_authors)
    return StructuredSearchSynapse(
        size=size,
        author_usernames=author_usernames,
        version=get_version(),
    )


def generate_structured_search_task(
    query_string: str = None,
    size: int = 5,
    sort_by: SortType = None,
    earlier_than: datetime = None,
    later_than: datetime = None,
    author_usernames: list = None,
) -> StructuredSearchSynapse:
    """
    Generates a structured search task for the validator to send to the miner.
    """

    # Randomly generate the query_string if not provided.
    if query_string is None:
        query_string = random_query()

    # Randomly select the earlier_than and later_than if not provided.
    if later_than is None:
        # 0.8 ratio to set the later_than or not
        if random.random() < 0.8:
            later_than = random_past_datetime()
        else:
            later_than = None

    # Note: do not set the earlier_than by default if it is not provided.

    return StructuredSearchSynapse(
        query_string=query_string,
        size=size,
        earlier_than_timestamp=(earlier_than.timestamp() if earlier_than else None),
        later_than_timestamp=(later_than.timestamp() if later_than else None),
        author_usernames=author_usernames,
        sort_by=sort_by,
        version=get_version(),
    )


def random_eth_denver_segments(
    eth_denver_dataset_dir,
    num_sources=3,
):
    dataset_path = Path(eth_denver_dataset_dir)

    files = random.sample(list(dataset_path.glob("*.json")), num_sources)
    segments = []
    for file in files:
        with open(file) as f:
            data = json.load(f)
            segments.append(data)
    return segments


def generate_question_from_eth_denver_segments(llm_client, segments):
    knowledge_text = ""
    for segment in segments:
        knowledge_text += (
            "Talk Title: "
            + segment["episode_title"]
            + "\n"
            + "Speaker: "
            + segment["speaker"]
            + "\n"
            + "Text: "
            + segment["text"]
            + "\n\n"
        )

    prompt = (
        "You are a crypto researcher, and you will be given a list of speaker transcript segments as your source of knowledge in ETH Denver 2024. "
        "Your job is to look for a question about the speaker and text that can be answered by this segment"
        "Transcript segments:\n\n"
    )
    prompt += knowledge_text
    prompt += (
        "Provide the question in less than 15 words. "
        "Please give the question text only, without any additional context or explanation."
    )

    bt.logging.debug(f"Prompt: {prompt}")

    try:
        output = llm_client.chat.completions.create(
            model="gpt-4-turbo",
            # response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=1.5,
            timeout=60,
        )

        bt.logging.debug(
            f"generation questions LLM response: {output.choices[0].message.content}"
        )
        bt.logging.debug(
            f"LLM usage: {output.usage}, finish reason: {output.choices[0].finish_reason}"
        )
        return output.choices[0].message.content
    except Exception as e:
        bt.logging.error(f"Error during LLM completion: {e}")
        bt.logging.debug(print_exception(type(e), e, e.__traceback__))



def generate_semantic_search_task(
    query_string: str,
    index_name: str = "eth_denver",
    size: int = 5,
    version: str = None,
) -> SemanticSearchSynapse:
    """
    Generates a semantic search task for the validator to send to the miner.
    """
    if version is None:
        version = get_version()

    return SemanticSearchSynapse(
        query_string=query_string,
        index_name=index_name,
        size=size,
        version=version,
    )


def find_repo(path):
    "Find repository root from the path's parents"
    for path in Path(path).parents:
        # Check whether "path/.git" exists and is a directory
        git_dir = path / ".git"
        if git_dir.is_dir():
            return path


# `python -m openkatio.tasks`
if __name__ == "__main__":
    # task = generate_structured_search_task("BTC")
    # print(task)
    # print(task.name)

    load_dotenv()
    llm_client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        organization=os.getenv("OPENAI_ORGANIZATION"),
        max_retries=3,
    )

    bt.logging.set_trace(True)
    repo_root = find_repo(__file__)
    eth_denver_dataset_dir = repo_root / "datasets/eth_denver_dataset"
    print(eth_denver_dataset_dir)
    print("generating question from ETH Denver dataset")
    segments = random_eth_denver_segments(eth_denver_dataset_dir, num_sources=3)
    question = generate_question_from_eth_denver_segments(llm_client, segments)
    print(question)

    task = generate_semantic_search_task(question)

    print(task)
