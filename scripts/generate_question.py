
if __name__ == "__main__":
    load_dotenv()

    search_client = Elasticsearch(
        os.environ["ELASTICSEARCH_HOST"],
        basic_auth=(
            os.environ["ELASTICSEARCH_USERNAME"],
            os.environ["ELASTICSEARCH_PASSWORD"],
        ),
        verify_certs=False,
        ssl_show_warn=False,
    )

    llm_client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        organization=os.getenv("OPENAI_ORGANIZATION"),
        max_retries=3,
    )

    docs = [
        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 9.997, "segment_end_time": 97.41,
         "text": "Hello everybody, welcome to ETH Denver 24. Great to see you, great to be here. It's always a good time, February, starting out ETH Denver. Ritual obviously. I'm Bernhard and this is Alex A. We're both with Fluence and we're here to introduce or discuss Fluence Cloudless Compute with Fluence in general and of course how it pertains to unlocking 25,000 USD through the hackathon. So we at Fluence believe that the cloud, as we know it, is broken. It's controlled by oligopolists that seek excessive rents, that want your trust, but at the same time don't give it. They have high exit barriers, technology lock-in, deplatforming approaches, and more. The way Fluence attempts to solve this problem is through a set of protocols. where we have decentralized serverless compute, we have decentralized physical infrastructure, and we have a decentralized compute marketplace that matches compute capacity from independent data center operator participating in the deep end structure with developers' requests and desire for capacity to run the serverless compute. And influence everything being decentralized, we verify, we don't trust. So from a use case perspective, obviously it's serverless, so you can do whatever you want with any other serverless.",
         "speaker": "Bernhard Borges", "segment_id": 0, "doc_id": "_aRTKs6AmvI.0"},

        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 97.57, "segment_end_time": 179.581,
         "text": "However, due to the proof-based cryptographic system underlying the Fluence Cloudless Compute, Verifiability at the runtime and execution layer is actually paramount and lends itself to some really interesting use cases, particularly around verifiable data preparation networks, real-time data pipelines, decentralized RPC solutions, and whatever you can imagine in your trustless mind. On the verifiable data side, it's interesting because as AI is getting increasingly regulated and you want provenance from data through data preparation to data lakes to AI models. Fluent serverless can really help you significantly more than any other of the traditional cloud providers can. From an implementation perspective, we don't have time to dig into all the protocol aspects. But the behavior of the protocol, which is off-chain compute, is actually carried out by what we call a reference peer. It's a Rust implementation called Knox. And it's built on libp2p. And it has a Marine Pool, which is a generalized WebAssembly runtime developed by Fluence, and an AquaVM Pool, which actually handles your distributed choreography of your functions. You can think of it as distributed workflows.",
         "speaker": "Bernhard Borges", "segment_id": 1, "doc_id": "_aRTKs6AmvI.1"},

        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 180.321, "segment_end_time": 229.884,
         "text": "And on top of it, you have a scheduler, basically cron jobs, that allows you to manage the event triggers. And from the network perspective, it's truly deep and bottom up. We have large data centers, mostly Filecoin miners right now, that provide significant capacity to the network for which they get paid if they're just providing capacity through a proof of capacity and or executing your functions for payment from the developer side. How do you create these functions? How do you go about it? You basically have your business logic. You code them in Rust. You end up with marine services, which is WebAssembly. And then you package your availability requirements. Do I want one invocation? Do I want one instance of my function?",
         "speaker": "Bernhard Borges", "segment_id": 2, "doc_id": "_aRTKs6AmvI.2"},
        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 229.904, "segment_end_time": 319.905,
         "text": "Do I want three? This is how you can build through our distributed choreography engine your own failover and high availability solution. Provider specifications, do you want to only operate with or work with green data centers, do you have geo preferences, and of course the willingness to pay for your services. On the other side, you have the providers, which they have their servers, they have peers, which we just saw, these NOXs, and then, spilling, we sliced capacity into what we call compute units, which actually are cores, for the most part, two-threaded cores, and I think right now for four gigabyte of RAM per CU allocation, which is what will run your functions. Providers give these capacity commitments to the chain and the chain is actually, it's Fluence's own blockchain built on IPC, Interplanetary Consensus blockchain from Protocol Labs. And they also provide their offers where they say, hey, this is what we offer, this is the kind of CPUs, this is the kind of RAM, this is our location, this is our green status, our carbon footprint. What developer attributes these are, the services we can provide, or we want to provide, and how much we actually need to pay or accept. And that gives an offer. And then the marketplace, which is on-chain, matches those offers from developers and providers.",
         "speaker": "Bernhard Borges", "segment_id": 3, "doc_id": "_aRTKs6AmvI.3"},

        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 320.065, "segment_end_time": 421.243,
         "text": "And then you create a deal, which then leads to deployment and the execution or the availability of your functions to be executed. Okay, I only got five minutes left to get to the bounty. So let me go through this quickly because Alexei is going to dive into some of the aspects a little bit deeper. We have five cloudless serverless compute tracks for a total of 25,000. We have 5,000 for each track. And hackers should be cognizant that A, you can provide or submit to more than one track. And you can actually integrate our challenges, if you will, our bounty tracks into any of your other hackathon projects over the week. So the first one, and it goes sort of from easy to a little bit harder along the way. The first one is basically just hit the documentation, hit some of the videos, and create, and employ, and execute a cloudless function. That's basically it. And for that, we have five times USDC 1,000. Improve your DAP with FRPC. So as most of you are aware, RPC is a sticky subject in the community. On the one hand, it's almost impossible for most app operators to run their own failover nodes. So you end up with RPC as a service, which is very, very available and very, very cheap. However, it has problems. It has single points of failure issues. It has privacy issues. It had data trust issues. And multiple solutions have emerged. And what we've done is we built on Fluence.",
         "speaker": "Bernhard Borges", "segment_id": 4, "doc_id": "_aRTKs6AmvI.4"},

        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 422.924, "segment_end_time": 524.871,
         "text": "Cloudless Compute, we built a substrate for you to basically orchestrate multiple RPC endpoints, service endpoints, Infura, Alchemy, many of them, into a decentralized solution to eliminate the major sticking points, which I just mentioned, including a single point of failure, all this good stuff. You work on that, we have that substrate ready. And that substrate consists of multiple components, including some Fluence Cloudless functions, but also some pretty interesting distributed algorithms on how you manage the workflow, the choreography of these services. Anyway, you manage to do this, you implement it and use it in your DAP, and you're in the running four or five times 1,000 USDC. Fluence Cloudless is stateless, fundamentally. So if you want persistence-particular data, you need to add that on your own. And one of the more interesting projects is Ceramic. And if you integrate Ceramic, Ceramic Stream, or ComposeDB into your Fluence Cloudless, you're in the running for, again, 5,000, 2,500, 1,500, 1,000 each. Now, this is my pet project, and I hope we're going to get a lot of participation in that one. With EIP-4844, it's interesting that you can use this blob as a sort of intermittent or intermediate, if you will, state persistent option. And anybody who wants to indulge me and use that in their hackathon project, go ahead.",
         "speaker": "Bernhard Borges", "segment_id": 5, "doc_id": "_aRTKs6AmvI.5"},

        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 525.491, "segment_end_time": 589.784,
         "text": "Again, you're in the running for 5,000 USDC, 2,500, 1,500, and 1,000. The last one is a little bit more involved and is more on the library enabling side. We give you two options, an MPC option and a ZK option. On the MPC option, basically what we want you to do is port a multiparty TSS crate, ECDSA, yeah, and demonstrate its use through orchestration with our orchestration language, which is Aqua. On the ZK side, port the Halo 2 crate, which already is close to WebAssembly, to Marine WebAssembly, and demonstrate its use by proving hamming distances. The ZK option is actually fairly, it's not as complicated as it seems since the circuit is, I gave you examples for that in the repo, is already, has been written. So you literally can copy that circuit and demonstrate this. So it's all about how to use it from Fluence as opposed to how to write the circuit. Again, 5,000, 2,500, 1,500, and 1,000. And that's it for me. So Alexei will take over, thanks.",
         "speaker": "Bernhard Borges", "segment_id": 6, "doc_id": "_aRTKs6AmvI.6"},

        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 596.596, "segment_end_time": 645.915,
         "text": "Thank you, Bernhard Borges. Hello, all. I'm Alex. We're going to go through the experience of developing apps for Fluence. And today, we're going to focus on Cloudless Functions and their architecture. You may already kind of know it because it resembles a lot of serverless architectures out there. For example, AWS Lambda. So it's pretty simple. You have compute functions on the right. They're usually written in Rust. They're accessible from the peer-to-peer network. And we have an HTTP Gateway, which is complex a little bit. And it can call those compute functions and can serve an HTTP API. So an HTTP client can call it. If we look a little bit deeper, then what happened? My laptop hang.",
         "speaker": "Alex A", "segment_id": 7, "doc_id": "_aRTKs6AmvI.7"},

        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 649.317, "segment_end_time": 649.557, "text": "No.",
         "speaker": "SPEAKER_00", "segment_id": 8, "doc_id": "_aRTKs6AmvI.8"},

        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 659.763, "segment_end_time": 746.185,
         "text": "Oops, sorry. I have no idea. Yeah, thanks. No, it's my laptop. It's not yours. So I guess there are going to be no demo. So the Cloudless functions are usually written, the compute functions are usually written in Rust and are accessible through the peer-to-peer network. So if we look inside the HTTP gateway, we will see that it actually consists of several components. Obviously, it consists of simple HTTP server. And it has a Cloudless function inside, which you could also call distribute workflow. It's written in Aqua language. And we have a Fluent JS client that lets you send those cloudless functions to the peer-to-peer network. And the cloudless function itself specifies the peer discovery mechanism to discover peers that host certain compute functions. In order to start developing Fluence CLI project, you will need to set up Fluence CLI project. And there are two ways to do that.",
         "speaker": "Alex A", "segment_id": 9, "doc_id": "_aRTKs6AmvI.9"},

        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 747.445, "segment_end_time": 827.923,
         "text": "One easy way to do just Fluent init here on the right, on the left in your case. On the down, the interactive wizard will guide you for all the steps. You will have to select the quick start or whatever, and just follow the instructions. Or if you're proficient enough with Fluent, you could do Fluent in it with all the things specified. So it will create a quick start project with the following structure. The most important things it contains are cloudless functions in the aqua directory, the HTTP gateway, which calls those cloudless functions. It contains compute functions definitions written in Rust. And it contains configuration files which configure your deployment. And if you're using local development environment, which we recommend you to do, then you will also configure it in provider.yaml. So let's see it in action. I guess not today. Yeah. So what it does, so under the hood of what happens in Fluence, you will find peer discovery.",
         "speaker": "Alex A", "segment_id": 10, "doc_id": "_aRTKs6AmvI.10"},

        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 828.223, "segment_end_time": 909.533,
         "text": "It works based on the fact that deal participants are stored on chain. So whenever you deploy something to the network, it actually finds, the process looks like that. So at first, you have deployment in your Fluence.yaml. You configure different stuff, like what's the replication factor? That's the target workers. That's how many peers will be chosen to deploy this function to. How much you would like to pay for a certain period of time for this function to be hosted and available. What's the initial balance of the deal? That means how long it will live until you have to put more money on it. and what services, what compute functions it includes. Once you have that, you run Fluent Deploy, and you get cloudless distributive uploaded to IPFS or to IPC. Effectively, you get CID back, which is sent to the chain, which creates a developer offer, which is then matched on chain Through the market, deals are matched with providers. So providers specify their prices, available APIs and stuff. Developers specify its desired price, desired APIs and stuff, and market matches it all.",
         "speaker": "Alex A", "segment_id": 11, "doc_id": "_aRTKs6AmvI.11"},

        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 910.054, "segment_end_time": 1020.961,
         "text": "And as a result, you receive an active deal. A deal is active once it has enough balance to go on, to burn through the money, to pay to providers, and once providers signal to the chain that they are ready to serve that function. So based on that information on chain, the peer discovery mechanism works by querying that information from chain and returning that information back to Aqua, which is your distributed workflow. It iterates through the peers in the subnet, in the deal, and is able to call the function on each of them, collect the answers, and present it to user or let it do whatever it wants to do, modify data process and stuff.",
         "speaker": "Alex A", "segment_id": 12, "doc_id": "_aRTKs6AmvI.12"},

        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 1049.659, "segment_end_time": 1133.457,
         "text": "All right, thanks. OK, so a lot of things Alex H has covered are actually things you as a developer don't have to take care of. This is all taken care of for you by both the CLI, which is our it's magic, but it's your way to scaffold and manage your project and by the various components including Marine in particular Aqua. So Aqua manages the topology so you don't have to program at the libp2p level at all and one of the interesting part that comes from those deals Alexei was talking about basically those deals when they're done where you match providers with developers and their requirements as well as the resource the peers you actually get something what we call a subnet which actually is an overlay network on the libp2p network that is managed through the configuration of your Aqua. So again, you don't have to chase down any of that the discovery is it's not inherent. It's not intrinsic but it's taken care of for you. So you can really focus on building your function and choreographing them into these cloudless applications you probably want to end up with. Now, I know this was really short, 20 minutes.",
         "speaker": "Bernhard Borges", "segment_id": 13, "doc_id": "_aRTKs6AmvI.13"},

        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 1133.577, "segment_end_time": 1183.962,
         "text": "It wasn't much time. So we have a much longer workshop on the 27th, 90 minutes. Starts at 1. And you'll just have to check our channels, including Discord, which is Fluence.chat, or our Twitter, which is Twitter.com slash Fluence underscore project for more details. But if you're interested, definitely come to that workshop. If you have questions in between now and then, hit us up on Discord or track us down in the building, and we're happy to help. We obviously want you to take home that money. Those USDC, that's our entire purpose of being here, for you to succeed. So if you want to take advantage of it, we're certainly up for it.",
         "speaker": "Bernhard Borges", "segment_id": 14, "doc_id": "_aRTKs6AmvI.14"},

        {"episode_id": "_aRTKs6AmvI",
         "episode_title": "Introduction to Decentralized Serverless Compute With Fluence Functions and Aqua | Bernhard Borges",
         "episode_url": "https://www.youtube.com/watch?v=_aRTKs6AmvI", "created_at": "2024-02-25T00:00:00.000Z",
         "company_name": "__OTHERS__", "segment_start_time": 1186.384, "segment_end_time": 1233.233,
         "text": "Okay, Alexei has one more thing. Can you switch me, please? I hope it won't crash again. Yeah, so just quick, I have like 30 seconds or something. So this is how compute functions look. It's just Rust code, right? It can do whatever it wants. It can access external resources through HTTP or whatever. But here it just returns a string. Here is what your Fluence YAML looks where you configure the development. And here you can see that it specifies that it uses the service that contains this compute function. And specifies the price all that and we can deploy it to the chain and it will work. That's how the project looks. That's how config looks. This is it.",
         "speaker": "Alex A", "segment_id": 15, "doc_id": "_aRTKs6AmvI.15"}
    ]

    for i, doc in enumerate(docs):
        segments = [doc]
        question = generate_question_from_eth_denver_segments(
            llm_client, segments
        )

        print(f"////////////////////////////////////////////////////////////////////////////////////")
        print(f"DOC:::{doc}")
        print(f"QUESTION:::{question}")
        print(f"////////////////////////////////////////////////////////////////////////////////////")