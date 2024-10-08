<div align="center">
    <h1>Awesome Foundation Model Leaderboard</h1>
    <a href="https://awesome.re">
        <img src="https://awesome.re/badge.svg" height="20"/>
    </a>
    <a href="https://github.com/SAILResearch/awesome-foundation-model-leaderboards/fork">
        <img src="https://img.shields.io/badge/PRs-Welcome-red" height="20"/>
    </a>
    <a href="https://arxiv.org/abs/2407.04065">
        <img src="https://img.shields.io/badge/📃-Arxiv-b31b1b" height="20"/>
    </a>
</div>

**Awesome Foundation Model Leaderboard** is a curated list of awesome foundation model leaderboards (for an explanation of what a leaderboard is, please refer to this [post](https://huggingface.co/docs/leaderboards/index)), along with various development tools and evaluation organizations according to [our survey](https://arxiv.org/abs/2407.04065):

<p align="center"><strong>On the Workflows and Smells of Leaderboard Operations (LBOps):<br>An Exploratory Study of Foundation Model Leaderboards</strong></p>

<p align="center"><a href="https://github.com/zhimin-z">Zhimin (Jimmy) Zhao</a>, <a href="https://abdulali.github.io">Abdul Ali Bangash</a>, <a href="https://www.filipecogo.pro">Filipe Roseiro Côgo</a>, <a href="https://mcis.cs.queensu.ca/bram.html">Bram Adams</a>, <a href="https://research.cs.queensu.ca/home/ahmed">Ahmed E. Hassan</a></p>

<p align="center"><a href="https://sail.cs.queensu.ca">Software Analysis and Intelligence Lab (SAIL)</a></p>

If you find this repository useful, please consider giving us a star :star: and citation:

```
@article{zhao2024workflows,
  title={On the Workflows and Smells of Leaderboard Operations (LBOps): An Exploratory Study of Foundation Model Leaderboards},
  author={Zhao, Zhimin and Bangash, Abdul Ali and C{\^o}go, Filipe Roseiro and Adams, Bram and Hassan, Ahmed E},
  journal={arXiv preprint arXiv:2407.04065},
  year={2024}
}
```

Additionally, we provide a [search toolkit](https://huggingface.co/spaces/zhiminy/awesome-foundation-model-leaderboard-search) that helps you quickly navigate through the leaderboards.

_If you want to contribute to this list (please do), welcome to [propose a pull request](https://github.com/SAILResearch/awesome-foundation-model-leaderboards/fork)._

_If you have any suggestions, critiques, or questions regarding this list, welcome to [raise an issue](https://github.com/SAILResearch/awesome-foundation-model-leaderboards/issues/new)._

Also, a leaderboard should be included if only:

* It is actively maintained.
* It is related to foundation models.

## Table of Contents
- [**Tools**](#tools)
- [**Organizations**](#organizations)
- [**Evaluations**](#evaluations)
    - [Model-oriented](#model-oriented)
        - [Comprehensive](#comprehensive)
        - [Text](#text)
        - [Image](#image)
        - [Code](#code)
        - [Math](#math)
        - [Video](#video)
        - [Agent](#agent)
        - [Audio](#audio)
        - [3D](#3d)
    - [Solution-oriented](#solution-oriented)
    - [Data-oriented](#data-oriented)
    - [Metric-oriented](#metric-oriented)
    - [Meta Leaderboard](#meta-leaderboard)



# Tools

| Name | Description |
| ---- | ----------- |
| [Demo Leaderboard](https://huggingface.co/spaces/demo-leaderboard-backend/leaderboard) | Demo leaderboard helps users easily deploy their leaderboards with a standardized template. |
| [Demo Leaderboard Backend](https://huggingface.co/spaces/demo-leaderboard-backend/backend) | Demo leaderboard backend helps users manage the leaderboard and handle submission requests, check [this](https://huggingface.co/docs/leaderboards/leaderboards/building_page) for details. |
| [Gradio Leaderboard](https://huggingface.co/spaces/demo-leaderboard-backend/leaderboard) | gradio_leaderboard helps users build fully functional and performant leaderboard demos with gradio. |
| [Leaderboard Explorer](https://huggingface.co/spaces/leaderboards/LeaderboardsExplorer) | Leaderboard Explorer helps users navigate the diverse range of leaderboards available on Hugging Face Spaces. |
| [Open LLM Leaderboard Renamer](https://huggingface.co/spaces/Weyaxi/open-llm-leaderboard-renamer) | open-llm-leaderboard-renamer helps users rename their models in Open LLM Leaderboard easily. |
| [Open LLM Leaderboard Results PR Opener](https://huggingface.co/spaces/Weyaxi/leaderboard-results-to-modelcard) | Open LLM Leaderboard Results PR Opener helps users showcase Open LLM Leaderboard results in their model cards. |
| [Open LLM Leaderboard Scraper](https://github.com/Weyaxi/scrape-open-llm-leaderboard) | Open LLM Leaderboard Scraper helps users scrape and export data from Open LLM Leaderboard. |
| [Progress Tracker](https://huggingface.co/spaces/andrewrreed/closed-vs-open-arena-elo) | This app visualizes the progress of proprietary and open-source LLMs over time as scored by the [LMSYS Chatbot Arena](https://lmarena.ai/?leaderboard). |



# Organizations

| Name | Description |
| ---- | ----------- |
| [Allen Institute for AI](https://leaderboard.allenai.org) | Allen Institute for AI is a non-profit research institute that aims to conduct high-impact AI research and engineering for the common good. |
| [Papers With Code](https://paperswithcode.com) | Papers With Code is a community-driven platform for learning about state-of-the-art research papers on machine learning. |


# Evaluations

## Model-oriented

### Comprehensive

| Name | Description |
| ---- | ----------- |
| [CompassRank](https://rank.opencompass.org.cn) | CompassRank is a platform to offer a comprehensive, objective, and neutral evaluation reference of foundation mdoels for the industry and research. |
| [FlagEval](https://flageval.baai.ac.cn/#/leaderboard) | FlagEval is a comprehensive platform for evaluating foundation models. |
| [GenAI-Arena](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena) | GenAI-Arena hosts the visual generation arena, where various vision models compete based on their performance in image generation, image edition, and video generation. |
| [Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm) | Holistic Evaluation of Language Models (HELM) is a reproducible and transparent framework for evaluating foundation models. |
| [nuScenes](https://www.nuscenes.org) | nuScenes enables researchers to study challenging urban driving situations using the full sensor suite of a real self-driving car. |
| [SuperCLUE](https://www.superclueai.com) | SuperCLUE is a series of benchmarks for evaluating Chinese foundation models. |

### Text

| Name | Description |
| ---- | ----------- |
| [ACLUE](https://github.com/isen-zhang/ACLUE) | ACLUE is an evaluation benchmark for ancient Chinese language comprehension. |
| [AIR-Bench](https://huggingface.co/spaces/AIR-Bench/leaderboard) | AIR-Bench is a benchmark to evaluate heterogeneous information retrieval capabilities of language models. |
| [AlignBench](https://llmbench.ai/align/data) | AlignBench is a multi-dimensional benchmark for evaluating LLMs' alignment in Chinese. |
| [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval) | AlpacaEval is an automatic evaluator designed for instruction-following LLMs. |
| [ANGO](https://huggingface.co/spaces/AngoHF/ANGO-Leaderboard) | ANGO is a generation-oriented Chinese language model evaluation benchmark. |
| [Arabic Tokenizers Leaderboard](https://huggingface.co/spaces/MohamedRashad/arabic-tokenizers-leaderboard) | Arabic Tokenizers Leaderboard compares the efficiency of LLMs in parsing Arabic in its different dialects and forms. |
| [Arena-Hard-Auto](https://github.com/lm-sys/arena-hard-auto) | Arena-Hard-Auto is a benchmark for instruction-tuned LLMs. |
| [Auto-Arena](https://huggingface.co/spaces/Auto-Arena/Leaderboard) | Auto-Arena is a benchmark in which various language model agents engage in peer-battles to evaluate their performance. |
| [BeHonest](https://gair-nlp.github.io/BeHonest/#leaderboard) | BeHonest is a benchmark to evaluate honesty - awareness of knowledge boundaries (self-knowledge), avoidance of deceit (non-deceptiveness), and consistency in responses (consistency) - in LLMs. |
| [BenBench](https://gair-nlp.github.io/benbench) | BenBench is a benchmark to evaluate the extent to which LLMs conduct verbatim training on the training set of a benchmark over the test set to enhance capabilities. |
| [BiGGen-Bench](https://huggingface.co/spaces/prometheus-eval/BiGGen-Bench-Leaderboard) | BiGGen-Bench is a comprehensive benchmark to evaluate LLMs across a wide variety of tasks. |
| [Biomedical Knowledge Probing Leaderboard](https://huggingface.co/spaces/CDT-BMAI-GP/biomed_probing_leaderboard) | Biomedical Knowledge Probing Leaderboard aims to track, rank, and evaluate biomedical factual knowledge probing results in LLMs. |
| [BotChat](https://botchat.opencompass.org.cn) | BotChat assesses the multi-round chatting capabilities of LLMs through a proxy task, evaluating whether two ChatBot instances can engage in smooth and fluent conversation with each other. |
| [C-Eval](https://cevalbenchmark.com/static/leaderboard.html) | C-Eval is a Chinese evaluation suite for LLMs. |
| [C-Eval Hard](https://cevalbenchmark.com/static/leaderboard.html) | C-Eval Hard is a more challenging version of C-Eval, which involves complex LaTeX equations and requires non-trivial reasoning abilities to solve. |
| [Capability leaderboard](https://github.com/bigai-nlco/LooGLE) | Capability leaderboard is a platform to evaluate long context understanding capabilties of LLMs. |
| [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub) | Chain-of-Thought Hub is a benchmark to evaluate the reasoning capabilities of LLMs. |
| [ChineseFactEval](https://github.com/GAIR-NLP/factool/tree/main/datasets/chinese) | ChineseFactEval is a factuality benchmark for Chinese LLMs. |
| [CLEM](https://huggingface.co/spaces/colab-potsdam/clem-leaderboard) | CLEM is a framework designed for the systematic evaluation of chat-optimized LLMs as conversational agents. |
| [CLiB](https://github.com/jeinlee1991/chinese-llm-benchmark) | CLiB is a benchmark to evaluate Chinese LLMs. |
| [CMMLU](https://github.com/haonan-li/CMMLU) | CMMLU is a Chinese benchmark to evaluate LLMs' knowledge and reasoning capabilities. |
| [CMB](https://cmedbenchmark.llmzoo.com/static/leaderboard.html) | CMB is a multi-level medical benchmark in Chinese. |
| [CMMLU](https://github.com/haonan-li/CMMLU) | CMMLU is a benchmark to evaluate the performance of LLMs in various subjects within the Chinese cultural context. |
| [CMMMU](https://cmmmu-benchmark.github.io/#leaderboard) | CMMMU is a benchmark to test the capabilities of multimodal models in understanding and reasoning across multiple disciplines in the Chinese context. |
| [CompMix](https://qa.mpi-inf.mpg.de/compmix) | CompMix is a benchmark for heterogeneous question answering. |
| [Compression Leaderboard](https://huggingface.co/spaces/eson/tokenizer-arena) | Compression Leaderboard is a platform to evaluate the compression performance of LLMs. |
| [CoTaEval](https://huggingface.co/spaces/boyiwei/CoTaEval_leaderboard) | CoTaEval is a benchmark to evaluate the feasibility and side effects of copyright takedown methods for LLMs. |
| [ConvRe](https://huggingface.co/spaces/3B-Group/ConvRe-Leaderboard) | ConvRe is a benchmark to evaluate LLMs' ability to comprehend converse relations. |
| [CriticBench](https://open-compass.github.io/CriticBench) | CriticBench is a benchmark to evaluate LLMs' ability to make critique responses. |
| [CRM LLM Leaderboard](https://huggingface.co/spaces/Salesforce/crm_llm_leaderboard) | CRM LLM Leaderboard is a platform to evaluate the efficacy of LLMs for business applications. |
| [DecodingTrust](https://decodingtrust.github.io/leaderboard) | DecodingTrust is an assessment platform to evaluate the trustworthiness of LLMs. |
| [Domain LLM Leaderboard](https://huggingface.co/spaces/NexaAIDev/domain_llm_leaderboard) | Domain LLM Leaderboard is a platform to evaluate the popularity of domain-specific LLMs. |
| [DyVal](https://llm-eval.github.io/pages/leaderboard/dyval.html) | DyVal is a dynamic evaluation protocol for LLMs. |
| [Enterprise Scenarios leaderboard](https://huggingface.co/spaces/PatronusAI/enterprise_scenarios_leaderboard) | Enterprise Scenarios Leaderboard aims to assess the performance of LLMs on real-world enterprise use cases. |
| [EQ-Bench](https://eqbench.com) | EQ-Bench is a benchmark to evaluate aspects of emotional intelligence in LLMs. |
| [Factuality Leaderboard](https://github.com/gair-nlp/factool) | Factuality Leaderboard compares the factual capabilities of LLMs. |
| [FuseReviews](https://huggingface.co/spaces/lovodkin93/FuseReviews-Leaderboard) | FuseReviews aims to advance grounded text generation tasks, including long-form question-answering and summarization. |
| [FELM](https://hkust-nlp.github.io/felm) | FELM is a meta benchmark to evaluate factuality evaluation benchmark for LLMs. |
| [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) | GAIA aims to test fundamental abilities that an AI assistant should possess. |
| [GPT-Fathom](https://github.com/GPT-Fathom/GPT-Fathom) | GPT-Fathom is an LLM evaluation suite, benchmarking 10+ leading LLMs as well as OpenAI's legacy models on 20+ curated benchmarks across 7 capability categories, all under aligned settings. |
| [Guerra LLM AI Leaderboard](https://huggingface.co/spaces/luisrguerra/guerra-llm-ai-leaderboard) | Guerra LLM AI Leaderboard compares and ranks the performance of LLMs across quality, price, performance, context window, and others. |
| [Hallucinations Leaderboard](https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard) | Hallucinations Leaderboard aims to track, rank and evaluate hallucinations in LLMs. |
| [HalluQA](https://github.com/OpenMOSS/HalluQA) | HalluQA is a benchmark to evaluate the phenomenon of hallucinations in Chinese LLMs. |
| [HellaSwag](https://rowanzellers.com/hellaswag) | HellaSwag is a benchmark to evaluate common-sense reasoning in LLMs. |
| [HHEM Leaderboard](https://huggingface.co/spaces/vectara/leaderboard) | HHEM Leaderboard evaluates how often a language model introduces hallucinations when summarizing a document. |
| [IFEval](https://huggingface.co/spaces/Krisseck/IFEval-Leaderboard) | IFEval is a benchmark to evaluate LLMs' instruction following capabilities with verifiable instructions. |
| [Indic LLM Leaderboard](https://huggingface.co/spaces/Cognitive-Lab/indic_llm_leaderboard) | Indic LLM Leaderboard is a benchmark to track progress and rank the performance of Indic LLMs. |
| [InstructEval](https://declare-lab.github.io/instruct-eval) | InstructEval is an evaluation suite to assess instruction selection methods in the context of LLMs. |
| [Japanese Chatbot Arena](https://huggingface.co/spaces/yutohub/japanese-chatbot-arena-leaderboard) | Japanese Chatbot Arena hosts the chatbot arena, where various LLMs compete based on their performance in Japanese. |
| [JustEval](https://allenai.github.io/re-align/just_eval.html) | JustEval is a powerful tool designed for fine-grained evaluation of LLMs. |
| [Ko Chatbot Arena](https://elo.instruct.kr/leaderboard) | Ko Chatbot Arena hosts the chatbot arena, where various LLMs compete based on their performance in Korean. |
| [KoLA](http://103.238.162.37:31622/LeaderBoard) | KoLA is a benchmark to evaluate the world knowledge of LLMs. |
| [L-Eval](https://l-eval.github.io) | L-Eval is a Long Context Language Model (LCLM) evaluation benchmark to evaluate the performance of handling extensive context. |
| [Language Model Council](https://llm-council.com) | Language Model Council (LMC) is a benchmark to evaluate tasks that are highly subjective and often lack majoritarian human agreement. |
| [LawBench](https://lawbench.opencompass.org.cn/leaderboard) | LawBench is a benchmark to evaluate the legal capabilities of LLMs. |
| [LogicKor](https://lawbench.opencompass.org.cn/leaderboard) | LogicKor is a benchmark to evaluate the multidisciplinary thinking capabilities of Korean LLMs. |
| [Long In-context Learning Leaderboard](https://huggingface.co/spaces/TIGER-Lab/LongICL-Leaderboard) | Long In-context Learning Leaderboard is a platform to evaluate long in-context learning evaluations for LLMs. |
| [LAiW](https://huggingface.co/spaces/daishen/SCULAiW) | LAiW is a benchmark to evaluate Chinese legal language understanding and reasoning. |
| [LLM Benchmarker Suite](https://llm-evals.formula-labs.com) | LLM Benchmarker Suite is a benchmark to evaluate the comprehensive capabilities of LLMs. |
| [LLM Leaderboard](https://huggingface.co/spaces/CathieDaDa/LLM_leaderboard) | LLM Leaderboard is a platform to evaluate LLMs in the Chinese context. |
| [LLM Leaderboard (en)](https://huggingface.co/spaces/CathieDaDa/LLM_leaderboard_en) | LLM Leaderboard (en) is a platform to evaluate LLMs in the English context. |
| [LLM Safety Leaderboard](https://huggingface.co/spaces/AI-Secure/llm-trustworthy-leaderboard) | LLM Safety Leaderboard aims to provide a unified evaluation for language model safety. |
| [LLM-Leaderboard](https://github.com/LudwigStumpp/llm-leaderboard) | LLM-Leaderboard is a joint community effort to create one central leaderboard for LLMs. |
| [LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard) | LLM-Perf Leaderboard aims to benchmark the performance of LLMs with different hardware, backends, and optimizations. |
| [LLMs Disease Risk Prediction Leaderboard](https://huggingface.co/spaces/TemryL/LLM-Disease-Risk-Leaderboard) | LLMs Disease Risk Prediction Leaderboard is a platform to evaluate LLMs on disease risk prediction. |
| [LLMEval](http://llmeval.com) | LLMEval is a benchmark to evaluate the quality of open-domain conversations with LLMs. |
| [LLMHallucination Leaderboard](https://huggingface.co/spaces/ramiroluo/LLMHallucination_Leaderboard) | Hallucinations Leaderboard evaluates LLMs based on an array of hallucination-related benchmarks. |
| [LLMPerf](https://github.com/ray-project/llmperf-leaderboard) | LLMPerf is a tool to evaluate the performance of LLMs using both load and correctness tests. |
| [LMSYS Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) | LMSYS Chatbot Arena Leaderboard hosts the chatbot arena, where various LLMs compete based on their performance in English. |
| [LongBench](https://github.com/THUDM/LongBench) | LongBench is a benchmark for assessing the long context understanding capabilities of LLMs. |
| [LucyEval](http://lucyeval.besteasy.com/leaderboard.html) | LucyEval offers a thorough assessment of LLMs' performance in various Chinese contexts. |
| [M3KE](https://github.com/tjunlp-lab/M3KE) | M3KE is a massive multi-level multi-subject knowledge evaluation benchmark to measure the knowledge acquired by Chinese LLMs. |
| [MINT](https://xwang.dev/mint-bench) | MINT is a benchmark to evaluate LLMs' ability to solve tasks with multi-turn interactions by using tools and leveraging natural language feedback. |
| [MedBench](https://medbench.opencompass.org.cn/leaderboard) | MedBench is a benchmark to evaluate the mastery of knowledge and reasoning abilities in medical LLMs. |
| [Meta Open LLM leaderboard](https://huggingface.co/spaces/felixz/meta_open_llm_leaderboard) | The Meta Open LLM leaderboard serves as a central hub for consolidating data from various open LLM leaderboards into a single, user-friendly visualization page. |
| [Mistral ChatBot Arena](https://huggingface.co/spaces/rwitz/Mistral-ChatBot-Arena) | Mistral ChatBot Arena hosts the chatbot arena, where various LLMs compete based on their performance in chatting. |
| [MixEval](https://mixeval.github.io/#leaderboard) | MixEval is a benchmark to evaluate LLMs via by strategically mixing off-the-shelf benchmarks. |
| [ML.ENERGY Leaderboard](https://ml.energy/leaderboard) | ML.ENERGY Leaderboard evaluates the energy consumption of LLMs. |
| [MMLU](https://github.com/hendrycks/test) | MMLU is a benchmark to evaluate the performance of LLMs across a wide array of natural language understanding tasks. |
| [MMLU-by-task Leaderboard](https://huggingface.co/spaces/CoreyMorris/MMLU-by-task-Leaderboard) | MMLU-by-task Leaderboard provides a platform for evaluating and comparing various ML models across different language understanding tasks. |
| [MMLU-Pro](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro) | MMLU-Pro is a more challenging version of MMLU to evaluate the reasoning capabilities of LLMs. |
| [ModelScope LLM Leaderboard](https://modelscope.cn/leaderboard/58/ranking?type=free) | ModelScope LLM Leaderboard is a platform to evaluate LLMs objectively and comprehensively. |
| [MSTEB](https://huggingface.co/spaces/clibrain/Spanish-Embeddings-Leaderboard) | MSTEB is a benchmark for measuring the performance of text embedding models in Spanish. |
| [MTEB](https://huggingface.co/spaces/mteb/leaderboard) | MTEB is a massive benchmark for measuring the performance of text embedding models on diverse embedding tasks across 112 languages. |
| [MY Malay LLM Leaderboard](https://huggingface.co/spaces/mesolitica/malay-llm-leaderboard) | MY Malay LLM Leaderboard aims to track, rank, and evaluate open LLMs on Malay tasks. |
| [MY Malaysian Embedding Leaderboard](https://huggingface.co/spaces/mesolitica/malaysian-embedding-leaderboard) | MY Malaysian Embedding Leaderboard measures and ranks the performance of text embedding models on diverse embedding tasks in Malay. |
| [NoCha](https://novelchallenge.github.io) | NoCha is a benchmark to evaluate how well long-context language models can verify claims written about fictional books. |
| [NPHardEval](https://huggingface.co/spaces/NPHardEval/NPHardEval-leaderboard) | NPHardEval is a benchmark to evaluate the reasoning abilities of LLMs through the lens of computational complexity classes. |
| [Occiglot Euro LLM Leaderboard](https://huggingface.co/spaces/occiglot/euro-llm-leaderboard) | Occiglot Euro LLM Leaderboard compares LLMs in terms of four main languages from the Okapi benchmark and Belebele (French, Italian, German, Spanish and Dutch). |
| [OlympicArena](https://gair-nlp.github.io/OlympicArena/#leaderboard) | OlympicArena is a benchmark to evaluate the advanced capabilities of LLMs across a broad spectrum of Olympic-level challenges. |
| [oobabooga](https://oobabooga.github.io/benchmark.html) | Oobabooga is a benchmark to perform repeatable performance tests of LLMs with the oobabooga web UI. |
| [Open-LLM-Leaderboard](https://huggingface.co/spaces/Open-Style/OSQ-Leaderboard) | Open-LLM-Leaderboard evaluates LLMs in terms of language understanding and reasoning by transitioning from multiple-choice questions (MCQs) to open-style questions. |
| [Open-source Model Fine-Tuning Leaderboard](https://predibase.com/fine-tuning-index) | Open-source Model Fine-Tuning Leaderboard is a platform to rank and showcase models that have been fine-tuned using open-source datasets or frameworks. |
| [OpenEval](http://openeval.org.cn/rank) | OpenEval is a multidimensional and open evaluation system to assess Chinese LLMs. |
| [OpenLLM Turkish leaderboard](https://huggingface.co/spaces/malhajar/OpenLLMTurkishLeaderboard) | OpenLLM Turkish leaderboard tracks progress and ranks the performance of LLMs in Turkish. |
| [Open Arabic LLM Leaderboard](https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard) | Open Arabic LLM Leaderboard tracks progress and ranks the performance of LLMs in Arabic. |
| [Open Dutch LLM Evaluation Leaderboard](https://huggingface.co/spaces/BramVanroy/open_dutch_llm_leaderboard) | Open Dutch LLM Evaluation Leaderboard tracks progress and ranks the performance of LLMs in Dutch. |
| [Open ITA LLM Leaderboard](https://huggingface.co/spaces/FinancialSupport/open_ita_llm_leaderboard) | Open ITA LLM Leaderboard tracks progress and ranks the performance of LLMs in Italian. |
| [Open Ko-LLM Leaderboard](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard) | Open Ko-LLM Leaderboard tracks progress and ranks the performance of LLMs in Korean. |
| [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) | Open LLM Leaderboard tracks progress and ranks the performance of LLMs in English. |
| [Open Medical-LLM Leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard) | Open Medical-LLM Leaderboard aims to track, rank, and evaluate open LLMs in the medical domain. |
| [Open MLLM Leaderboard](https://github.com/hkust-nlp/felm) | Open MLLM Leaderboard aims to track, rank and evaluate LLMs and chatbots. |
| [Open MOE LLM Leaderboard](https://huggingface.co/spaces/sparse-generative-ai/open-moe-llm-leaderboard) | OPEN MOE LLM Leaderboard assesses the performance and efficiency of various Mixture of Experts (MoE) LLMs. |
| [Open Multilingual LLM Evaluation Leaderboard](https://huggingface.co/spaces/uonlp/open_multilingual_llm_leaderboard) | Open Multilingual LLM Evaluation Leaderboard tracks progress and ranks the performance of LLMs in multiple languages. |
| [Open PL LLM Leaderboard](https://github.com/hkust-nlp/felm) | Open PL LLM Leaderboard is a platform for assessing the performance of various LLMs in Polish. |
| [Open PT LLM Leaderboard](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard) | Open PT LLM Leaderboard tracks progress and ranks the performance of LLMs in Portuguese. |
| [OR-Bench](https://huggingface.co/spaces/bench-llms/or-bench-leaderboard) | OR-Bench is a benchmark to evaluate the over-refusal of enhanced safety in LLMs. |
| [Powered-by-Intel LLM Leaderboard](https://huggingface.co/spaces/Intel/powered_by_intel_llm_leaderboard) | Powered-by-Intel LLM Leaderboard evaluates, scores, and ranks LLMs that have been pre-trained or fine-tuned on Intel Hardware. |
| [PubMedQA](https://pubmedqa.github.io) | PubMedQA is a benchmark to evaluate biomedical research question answering. |
| [PromptBench](https://llm-eval.github.io/pages/leaderboard/advprompt.html) | PromptBench is a benchmark to evaluate the robustness of LLMs on adversarial prompts. |
| [QuALITY](https://nyu-mll.github.io/quality) | QuALITY is a benchmark for evaluating multiple-choice question-answering with a long context. |
| [RABBITS](https://huggingface.co/spaces/AIM-Harvard/rabbits-leaderboard) | RABBITS is a benchmark to evaluate the robustness of LLMs by evaluating their handling of synonyms, specifically brand and generic drug names. |
| [RedTeam Arena](https://redarena.ai/leaderboard) | RedTeam Arena is a red-teaming platform for LLMs. |
| [Red Teaming Resistance Benchmark](https://huggingface.co/spaces/HaizeLabs/red-teaming-resistance-benchmark) | Red Teaming Resistance Benchmark is a benchmark to evaluate the robustness of LLMs against red teaming prompts. |
| [Reviewer Arena](https://huggingface.co/spaces/openreviewer/reviewer-arena) | Reviewer Arena hosts the reviewer arena, where various LLMs compete based on their performance in critiquing academic papers. |
| [Robust Reading Competition](https://rrc.cvc.uab.es) | Robust Reading refers to the research area on interpreting written communication in unconstrained settings. |
| [RoleEval](https://github.com/magnetic2014/roleeval) | RoleEval is a bilingual benchmark to evaluate the memorization, utilization, and reasoning capabilities of role knowledge of LLMs. |
| [Safety Prompts](http://coai.cs.tsinghua.edu.cn/leaderboard) | Safety Prompts is a benchmark to evaluate the safety of Chinese LLMs. |
| [SafetyBench](https://llmbench.ai/safety/data) | SafetyBench is a benchmark to evaluate the safety of LLMs. |
| [SALAD-Bench](https://huggingface.co/spaces/OpenSafetyLab/Salad-Bench-Leaderboard) | SALAD-Bench is a benchmark for evaluating the safety and security of LLMs. |
| [ScandEval](https://scandeval.com) | ScandEval is a benchmark to evaluate LLMs on tasks in Scandinavian languages as well as German, Dutch, and English. |
| [SciKnowEval](http://scimind.ai/sciknoweval) | SciKnowEval is a benchmark to evaluate LLMs based on their proficiency in studying extensively, enquiring earnestly, thinking profoundly, discerning clearly, and practicing assiduously. |
| [SCROLLS](https://www.scrolls-benchmark.com/leaderboard) | SCROLLS is a benchmark to evaluate the reasoning capabilities of LLMs over long texts. |
| [SeaExam](https://huggingface.co/spaces/SeaLLMs/SeaExam_leaderboard) | SeaExam is a benchmark to evaluate LLMs for Southeast Asian (SEA) languages. |
| [SEAL](https://scale.com/leaderboard) | SEAL is an expert-driven private evaluation platform for LLMs. |
| [SeaEval](https://huggingface.co/spaces/SeaEval/SeaEval_Leaderboard) | SeaEval is a benchmark to evaluate the performance of multilingual LLMs in understanding and reasoning with natural language, as well as comprehending cultural practices, nuances, and values. |
| [Spec-Bench](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md) | Spec-Bench is a benchmark to evaluate speculative decoding methods across diverse scenarios. |
| [SuperBench](https://fm.ai.tsinghua.edu.cn/superbench) | SuperBench is a comprehensive evaluation system of tasks and dimensions to assess the overall capabilities of LLMs. |
| [SuperGLUE](https://super.gluebenchmark.com/leaderboard) | SuperGLUE is a benchmark to evaluate the performance of LLMs on a set of challenging language understanding tasks. |
| [SuperLim](https://lab.kb.se/leaderboard/results) | SuperLim is a benchmark to evaluate the language understanding capabilities of LLMs in Swedish. |
| [Swahili LLM-Leaderboard](https://github.com/msamwelmollel/Swahili_LLM_Leaderboard) | Swahili LLM-Leaderboard is a joint community effort to create one central leaderboard for LLMs. |
| [T-Eval](https://open-compass.github.io/T-Eval/leaderboard.html) | T-Eval is a benchmark for evaluating the tool utilization capability of LLMs. |
| [TAT-DQA](https://nextplusplus.github.io/TAT-DQA) | TAT-DQA is a benchmark to evaluate LLMs on the discrete reasoning over documents that combine both structured and unstructured information. |
| [TAT-QA](https://nextplusplus.github.io/TAT-QA) | TAT-QA is a benchmark to evaluate LLMs on the discrete reasoning over documents that combines both tabular and textual content. |
| [The Pile](https://pile.eleuther.ai) | The Pile is a benchmark to evaluate the world knowledge and reasoning ability of LLMs. |
| [TOFU Leaderboard](https://huggingface.co/spaces/locuslab/tofu_leaderboard) | TOFU Leaderboard is a benchmark to evaluate the unlearning performance of LLMs in realistic scenarios. |
| [Science Leaderboard](https://huggingface.co/spaces/wenhu/Science-Leaderboard) | Science Leaderboard is a platform to evaluate LLMs' capabilities to solve science problems. |
| [Toloka LLM Leaderboard](https://huggingface.co/spaces/toloka/open-llm-leaderboard) | Toloka LLM Leaderboard is a benchmark to evaluate LLMs based on authentic user prompts and expert human evaluation. |
| [Toolbench](https://huggingface.co/spaces/qiantong-xu/toolbench-leaderboard) | ToolBench is a platform for training, serving, and evaluating LLMs specifically for tool learning. |
| [Toxicity Leaderboard](https://huggingface.co/spaces/Bias-Leaderboard/leaderboard) | Toxicity Leaderboard evaluates the toxicity of LLMs. |
| [Trustbit LLM Leaderboards](https://www.trustbit.tech/en/llm-benchmarks) | Trustbit LLM Leaderboards is a platform that provides benchmarks for building and shipping products with LLMs. |
| [TrustLLM](https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html) | TrustLLM is a benchmark to evaluate the trustworthiness of LLMs. |
| [UGI Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard) | UGI Leaderboard measures and compares the uncensored and controversial information known by LLMs. |
| [ViDoRe](https://huggingface.co/spaces/vidore/vidore-leaderboard) | ViDoRe is a benchmark to evaluate retrieval models on their capacity to match queries to relevant documents at the page level. |
| [VLLMs Leaderboard](https://huggingface.co/spaces/vlsp-2023-vllm/VLLMs-Leaderboard) | VLLMs Leaderboard aims to track, rank and evaluate open LLMs and chatbots. |
| [Xiezhi](https://github.com/MikeGu721/XiezhiBenchmark) | Xiezhi is a benchmark for holistic domain knowledge evaluation of LLMs. |
| [Yet Another LLM Leaderboard](https://huggingface.co/spaces/mlabonne/Yet_Another_LLM_Leaderboard) | Yet Another LLM Leaderboard is a platform for tracking, ranking, and evaluating open LLMs and chatbots. |

## Image

| Name | Description |
| ---- | ----------- |
| [AesBench](https://github.com/yipoh/AesBench) | AesBench is a benchmark to evaluate multimodal LLMs (MLLM) on image aesthetics perception. |
| [BLINK](https://github.com/zeyofu/BLINK_Benchmark) | BLINK is a benchmark to evaluate the core visual perception abilities of MLLMs. |
| [CCBench](https://mmbench.opencompass.org.cn/leaderboard) | CCBench is a benchmark to evaluate the multi-modal capabilities of MLLMs specifically related to Chinese culture. |
| [CharXiv](https://charxiv.github.io/#leaderboard) | CharXiv is a benchmark to evaluate chart understanding capabilities of MLLMs. |
| [ChEF](https://openlamm.github.io/Leaderboards) | ChEF is a benchmark to evaluate MLLMs across various visual reasoning tasks. |
| [ConTextual](https://huggingface.co/spaces/ucla-contextual/contextual_leaderboard) | ConTextual is a benchmark to evaluate MLLMs across context-sensitive text-rich visual reasoning tasks. |
| [CORE-MM](https://core-mm.github.io) | CORE-MM is a benchmark to evaluate the open-ended visual question-answering (VQA) capabilities of MLLMs. |
| [DreamBench++](https://dreambenchplus.github.io/#leaderboard) | DreamBench++ is a human-aligned benchmark automated by multimodal models for personalized image generation. |
| [EgoPlan-Bench](https://huggingface.co/spaces/ChenYi99/EgoPlan-Bench_Leaderboard) | EgoPlan-Bench is a benchmark to evaluate planning abilities of MLLMs in real-world, egocentric scenarios. |
| [GlitchBench](https://huggingface.co/spaces/glitchbench/Leaderboard) | GlitchBench is a benchmark to evaluate the reasoning capabilities of MLLMs in the context of detecting video game glitches. |
| [HallusionBench](https://github.com/tianyi-lab/HallusionBench) | HallusionBench is a benchmark to evaluate the image-context reasoning capabilities of MLLMs. |
| [InfiMM-Eval](https://infimm.github.io/InfiMM-Eval) | InfiMM-Eval is a benchmark to evaluate the open-ended VQA capabilities of MLLMs. |
| [LRVS-Fashion](https://huggingface.co/spaces/Slep/LRVSF-Leaderboard) | LRVS-Fashion is a benchmark to evaluate LLMs regarding image similarity search in fashion. |
| [LVLM Leaderboard](https://github.com/OpenGVLab/Multi-Modality-Arena) | LVLM Leaderboard is a platform to evaluate the visual reasoning capabilities of MLLMs. |
| [M3CoT](https://lightchen233.github.io/m3cot.github.io/leaderboard.html) | M3CoT is a benchmark for multi-domain multi-step multi-modal chain-of-thought of MLLMs. |
| [Mementos](https://mementos-bench.github.io/#leaderboard) | Mementos is a benchmark to evaluate the reasoning capabilities of MLLMs over image sequences. |
| [MJ-Bench](https://huggingface.co/spaces/MJ-Bench/MJ-Bench-Leaderboard) | MJ-Bench is a benchmark to evaluate multimodal judges in providing feedback for image generation models across four key perspectives: alignment, safety, image quality, and bias. |
| [MLLM-Bench](https://mllm-bench.llmzoo.com/static/leaderboard.html) | MLLM-Bench is a benchmark to evaluate the visual reasoning capabilities of MLVMs. |
| [MMBench](https://mmbench.opencompass.org.cn/leaderboard) | MMBench is a benchmark to evaluate the visual reasoning capabilities of MLLMs. |
| [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) | MME is a benchmark to evaluate the visual reasoning capabilities of MLLMs. |
| [MMMU](https://mmmu-benchmark.github.io/#leaderboard) | MMMU is a benchmark to evaluate the performance of multimodal models on tasks that demand college-level subject knowledge and expert-level reasoning across various disciplines. |
| [MMStar](https://mmstar-benchmark.github.io/#Leaderboard) | MMStar is a benchmark to evaluate the multi-modal capacities of MLLMs. |
| [MMT-Bench](https://mmt-bench.github.io/#leaderboard) | MMT-Bench is a benchmark to evaluate MLLMs across a wide array of multimodal tasks that require expert knowledge as well as deliberate visual recognition, localization, reasoning, and planning. |
| [Multimodal Hallucination Leaderboard](https://huggingface.co/spaces/scb10x/multimodal-hallucination-leaderboard) | Multimodal Hallucination Leaderboard compares MLLMs based on hallucination levels in various tasks. |
| [MULTI](https://github.com/OpenDFM/MULTI-Benchmark) | MULTI is a benchmark to evaluate MLLMs on understanding complex tables and images, and reasoning with long context. |
| [MultiTrust](https://multi-trust.github.io/#leaderboard) | MultiTrust is a benchmark to evaluate the trustworthiness of MLLMs across five primary aspects: truthfulness, safety, robustness, fairness, and privacy. |
| [NPHardEval4V](https://github.com/lizhouf/nphardeval4v) | NPHardEval4V is a benchmark to evaluate the reasoning abilities of MLLMs through the lens of computational complexity classes. |
| [OCRBench](https://huggingface.co/spaces/echo840/ocrbench-leaderboard) | OCRBench is a benchmark to evaluate the OCR capabilities of multimodal models. |
| [Open CoT Leaderboard](https://huggingface.co/spaces/logikon/open_cot_leaderboard) | Open CoT Leaderboard tracks LLMs' abilities to generate effective chain-of-thought reasoning traces. |
| [Open Parti Prompts Leaderboard](https://huggingface.co/spaces/OpenGenAI/parti-prompts-leaderboard) | Open Parti Prompts Leaderboard compares text-to-image models to each other according to human preferences. |
| [PCA-Bench](https://github.com/pkunlp-icler/PCA-EVAL) | PCA-Bench is a benchmark to evaluate the embodied decision-making capabilities of multimodal models. |
| [Q-Bench](https://huggingface.co/spaces/q-future/Q-Bench-Leaderboard) | Q-Bench is a benchmark to evaluate the visual reasoning capabilities of MLLMs. |
| [RewardBench](https://huggingface.co/spaces/allenai/reward-bench) | RewardBench is a benchmark to evaluate the capabilities and safety of reward models. |
| [ScienceQA](https://scienceqa.github.io/leaderboard.html) | ScienceQA is a benchmark used to evaluate the multi-hop reasoning ability and interpretability of AI systems in the context of answering science questions. |
| [SciGraphQA](https://github.com/findalexli/SciGraphQA) | SciGraphQA is a benchmark to evaluate the MLLMs in scientific graph question-answering. |
| [SEED-Bench](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) | SEED-Bench is a benchmark to evaluate the text and image generation of multimodal models. |
| [UnlearnCanvas](https://huggingface.co/spaces/OPTML-Group/UnlearnCanvas-Benchmark) | UnlearnCanvas is a stylized image benchmark to evaluate machine unlearning for diffusion models. |
| [UnlearnDiffAtk](https://huggingface.co/spaces/Intel/UnlearnDiffAtk-Benchmark) | UnlearnDiffAtk is a benchmark to evaluate the robustness of safety-driven unlearned diffusion models (DMs) (i.e., DMs after unlearning undesirable concepts, styles, or objects) across a variety of tasks. |
| [URIAL Bench](https://huggingface.co/spaces/allenai/URIAL-Bench) | URIAL Bench is a benchmark to evaluate the capacity of language models for alignment without introducing the factors of fine-tuning (learning rate, data, etc.), which are hard to control for fair comparisons. |
| [UPD Leaderboard](https://huggingface.co/spaces/MM-UPD/MM-UPD_Leaderboard) | UPD Leaderboard is a platform to evaluate the trustworthiness of MLLMs in unsolvable problem detection. |
| [Vibe-Eval](https://github.com/reka-ai/reka-vibe-eval) | Vibe-Eval is a benchmark to evaluate MLLMs for challenging cases. |
| [VideoHallucer](https://videohallucer.github.io) | VideoHallucer is a benchmark to detect hallucinations in MLLMs. |
| [VisIT-Bench](https://visit-bench.github.io) | VisIT-Bench is a benchmark to evaluate the instruction-following capabilities of MLLMs for real-world use. |
| [Waymo Open Dataset Challenges](https://waymo.com/open/challenges) | Waymo Open Dataset Challenges hold diverse self-driving datasets to evaluate ML models. |
| [WHOOPS!](https://huggingface.co/spaces/nlphuji/WHOOPS-Leaderboard-Full) | WHOOPS! is a benchmark to evaluate the visual commonsense reasoning abilities of MLLMs. |
| [WildBench](https://huggingface.co/spaces/allenai/WildBench) | WildBench is a benchmark for evaluating language models on challenging tasks that closely resemble real-world applications. |
| [WildVision Arena Leaderboard](https://huggingface.co/spaces/WildVision/vision-arena) | WildVision Arena Leaderboard hosts the chatbot arena, where various MLLMs compete based on their performance in visual understanding. |

### Code

| Name | Description |
| ---- | ----------- |
| [Aider LLM Leaderboards](https://aider.chat/docs/leaderboards) | Aider LLM Leaderboards evaluate LLM's ability to follow system prompts to edit code. |
| [Berkeley Function Calling Leaderboard](https://huggingface.co/spaces/gorilla-llm/berkeley-function-calling-leaderboard) | Berkeley Function Calling Leaderboard evaluates the ability of LLMs to call functions (also known as tools) accurately. |
| [BigCodeBench](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard) | BigCodeBench is a benchmark for code generation with practical and challenging programming tasks. |
| [Big Code Models Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard) | Big Code Models Leaderboard to assess the performance of LLMs on code-related tasks. |
| [BIRD](https://bird-bench.github.io) | BIRD is a benchmark to evaluate the performance of text-to-SQL parsing systems. |
| [CanAiCode Leaderboard](https://huggingface.co/spaces/mike-ravkine/can-ai-code-results) | CanAiCode Leaderboard is a platform to assess the code generation capabilities of LLMs. |
| [ClassEval](https://fudanselab-classeval.github.io/leaderboard.html) | ClassEval is a benchmark to evaluate LLMs on class-level code generation. |
| [Code Lingua](https://codetlingua.github.io/leaderboard.html) | Code Lingua is a benchmark to compare the ability of code models to understand what the code implements in source languages and translate the same semantics in target languages. |
| [Coding LLMs Leaderboard](https://leaderboard.tabbyml.com) | Coding LLMs Leaderboard is a platform to evaluate and rank LLMs across various programming tasks. |
| [CRUXEval](https://crux-eval.github.io/leaderboard.html) | CRUXEval is a benchmark to evaluate code reasoning, understanding, and execution capabilities of LLMs. |
| [CyberSafetyEval](https://huggingface.co/spaces/facebook/CyberSecEval) | CYBERSECEVAL is a benchmark to evaluate the cybersecurity of LLMs as coding assistants. |
| [DevOps-Eval](https://github.com/codefuse-ai/codefuse-devops-eval) | DevOps-Eval is a benchmark to evaluate code models in the DevOps/AIOps field. |
| [DS-1000](https://ds1000-code-gen.github.io/model_DS1000.html) | DS-1000 is a meta benchmark to evaluate code generation models in data science tasks. |
| [EffiBench](https://huggingface.co/spaces/EffiBench/effibench-leaderboard) | EffiBench is a benchmark to evaluate the efficiency of LLMs in code generation. |
| [EvalPlus](https://evalplus.github.io/leaderboard.html) | EvalPlus is a benchmark to evaluate the code generation performance of LLMs. |
| [EvoCodeBench](https://github.com/seketeam/EvoCodeBench) | EvoCodeBench is an evolutionary code generation benchmark aligned with real-world code repositories. |
| [EvoEval](https://evo-eval.github.io/leaderboard.html) | EvoEval is a benchmark to evaluate the coding abilities of LLMs, created by evolving existing benchmarks into different targeted domains. |
| [BigCodeBench](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard) | BigCodeBench is a benchmark for code generation with practical and challenging programming tasks. |
| [InfiBench](https://infi-coder.github.io/infibench) | InfiBench is a benchmark to evaluate code models on answering freeform real-world code-related questions. |
| [InterCode](https://intercode-benchmark.github.io) | InterCode is a benchmark to standardize and evaluate interactive coding with execution feedback. || [Julia LLM Leaderboard](https://github.com/svilupp/Julia-LLM-Leaderboard) | Julia LLM Leaderboard is a platform to compare code models' abilities in generating syntactically correct Julia code, featuring structured tests and automated evaluations for easy and collaborative benchmarking. |
| [LiveCodeBench](https://huggingface.co/spaces/livecodebench/leaderboard) | LiveCodeBench is a benchmark to evaluate code models across code-related scenarios over time. |
| [Long Code Arena](https://huggingface.co/spaces/JetBrains-Research/long-code-arena) | Long Code Arena is a suite of benchmarks for code-related tasks with large contexts, up to a whole code repository. |
| [NaturalCodeBench](https://github.com/THUDM/NaturalCodeBench) | NaturalCodeBench is a benchmark to mirror the complexity and variety of scenarios in real coding tasks. |
| [Nexus Function Calling Leaderboard](https://huggingface.co/spaces/Nexusflow/Nexus_Function_Calling_Leaderboard) | Nexus Function Calling Leaderboard is a platform to evaluate code models on performing function calling and API usage.
| [Program Synthesis Models Leaderboard](https://accubits.com/open-source-program-synthesis-models-leaderboard) | Program Synthesis Models Leaderboard provides a ranking and comparison of open-source code models based on their performance. |
| [RepoQA](https://evalplus.github.io/repoqa.html) | RepoQA is a benchmark to evaluate the long-context code understanding ability of LLMs.
| [Spider](https://yale-lily.github.io/spider) | Spider is a benchmark to evaluate the performance of natural language interfaces for cross-domain databases. |
| [StableToolBench](https://huggingface.co/spaces/stabletoolbench/Stable_Tool_Bench_Leaderboard) | StableToolBench is a benchmark to evaluate tool learning that aims to provide a well-balanced combination of stability and reality. |
| [SWE-bench](https://www.swebench.com) | SWE-bench is a benchmark for evaluating LLMs on real-world software issues collected from GitHub. |

### Math

| Name | Description |
| ---- | ----------- |
| [MathBench](https://open-compass.github.io/MathBench) | MathBench is a multi-level difficulty mathematics evaluation benchmark for LLMs. |
| [MathEval](https://matheval.ai/leaderboard) | MathEval is a benchmark to evaluate the mathematical capabilities of LLMs. |
| [MathVerse](https://mathverse-cuhk.github.io/#leaderboard) | MathVerse is a benchmark to evaluate vision-language models in interpreting and reasoning with visual information in mathematical problems. |
| [MathVista](https://mathvista.github.io/#leaderboard) | MathVista is a benchmark to evaluate mathematical reasoning in visual contexts. |
| [Open Multilingual Reasoning Leaderboard](https://huggingface.co/spaces/kevinpro/Open-Multilingual-Reasoning-Leaderboard) | Open Multilingual Reasoning Leaderboard tracks and ranks the reasoning performance of LLMs on multilingual mathematical reasoning benchmarks. |
| [SciBench](https://scibench-ucla.github.io/#leaderboard) | SciBench is a benchmark to evaluate the reasoning capabilities of LLMs for solving complex scientific problems. |
| [TabMWP](https://promptpg.github.io/leaderboard.html) | TabMWP is a benchmark to evaluate LLMs in mathematical reasoning tasks that involve both textual and tabular data. |
| [We-Math](https://promptpg.github.io/leaderboard.html) | We-Math is a benchmark to evaluate the human-like mathematical reasoning capabilities of LLMs with problem-solving principles beyond the end-to-end performance. |

### Video

| Name | Description |
| ---- | ----------- |
| [AutoEval-Video](https://huggingface.co/spaces/khhuiyh/AutoEval-Video_LeaderBoard) | AutoEval-Video is a benchmark to evaluate the capabilities of video models in the context of open-ended video question answering. |
| [LongVideoBench](https://huggingface.co/spaces/longvideobench/LongVideoBench) | LongVideoBench is a benchmark to evaluate the capabilities of video models in answering referred reasoning questions, which are dependent on long frame inputs and cannot be well-addressed by a single frame or a few sparse frames. |
| [MLVU](https://github.com/JUNJIE99/MLVU) | MLVU is a benchmark to evaluate video models in multi-task long video understanding. |
| [MMToM-QA](https://chuanyangjin.com/mmtom-qa-leaderboard) | MMToM-QA is a multimodal benchmark to evaluate machine Theory of Mind (ToM), the ability to understand people's minds. |
| [MVBench](https://huggingface.co/spaces/OpenGVLab/MVBench_Leaderboard) | MVBench is a benchmark to evaluate the temporal understanding capabilities of video models in dynamic video tasks. |
| [VBench](https://vchitect.github.io/VBench-project) | VBench is a benchmark to evaluate video generation capabilities of video models. |
| [Video-Bench](https://huggingface.co/spaces/LanguageBind/Video-Bench) | Video-Bench is a benchmark to evaluate the video-exclusive understanding, prior knowledge incorporation, and video-based decision-making abilities of video models. |
| [Video-MME](https://video-mme.github.io/home_page.html#leaderboard) | Video-MME is a benchmark to evaluate the video analysis capabilities of video models. |
| [VNBench](https://videoniah.github.io/#leaderboard) | VNBench is a benchmark to evaluate the fine-grained understanding and spatio-temporal modeling capabilities of video models. |

### Agent

| Name | Description |
| ---- | ----------- |
| [Agent CTF Leaderboard](https://huggingface.co/spaces/autogenCTF/agent_ctf_leaderboard) | Agent CTF Leaderboard is a platform to evaluate the performance of LLM-driven agents in the field of cybersecurity, particularly CTF (capture the flag) competition issues. |
| [AgentBench](https://llmbench.ai/agent/data) | AgentBench is the benchmark to evaluate language model-as-Agent across a diverse spectrum of different environments. |
| [AgentStudio](https://skyworkai.github.io/agent-studio) | AgentStudio is an integrated solution featuring in-depth benchmark suites, realistic environments, and comprehensive toolkits. |
| [LLM Colosseum Leaderboard](https://github.com/OpenGenerativeAI/llm-colosseum) | LLM Colosseum Leaderboard is a platform to evaluate LLMs by fighting in Street Fighter 3. |
| [TravelPlanner](https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard) | TravelPlanner is a benchmark to evaluate LLM agents in tool use and complex planning within multiple constraints. |
| [VisualWebArena](https://jykoh.com/vwa) | VisualWebArena is a benchmark to evaluate the performance of multimodal web agents on realistic visually grounded tasks. |
| [WebArena](https://github.com/web-arena-x/webarena) | WebArena is a standalone, self-hostable web environment to evaluate autonomous agents. |

### Audio

| Name | Description |
| ---- | ----------- |
| [MY Malaysian Speech-to-Text Leaderboard](https://huggingface.co/spaces/mesolitica/malaysian-stt-leaderboard) | MY Malaysian Speech-to-Text (STT) Leaderboard aims to track, rank and evaluate Malaysian STT models. |
| [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) | Open ASR Leaderboard provides a platform for tracking, ranking, and evaluating Automatic Speech Recognition (ASR) models. |
| [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) | TTS-Arena hosts the Text To Speech (TTS) arena, where various TTS models compete based on their performance in generating speech. |

### 3D

| Name | Description |
| ---- | ----------- |
| [3D Arena](https://huggingface.co/spaces/dylanebert/3d-arena) | 3D Arena hosts 3D generation arena, where various 3D generative models compete based on their performance in generating 3D models. |
| [3D-POPE](https://huggingface.co/spaces/sled-umich/3D-POPE-leaderboard) | 3D-POPE is a benchmark to evaluate object hallucination in 3D generative models. |
| [3DGen-Arena](https://huggingface.co/spaces/ZhangYuhan/3DGen-Arena) | 3DGen-Arena hosts the 3D generation arena, where various 3D generative models compete based on their performance in generating 3D models. |
| [BOP](https://bop.felk.cvut.cz/leaderboards) | BOP is a benchmark for 6D pose estimation of a rigid object from a single RGB-D input image. |
| [GPTEval3D Leaderboard](https://huggingface.co/spaces/GPTEval3D/Leaderboard_dev) | GPTEval3D Leaderboard check how MLLMs understand 3D content via multi-view images as input. |

## Solution-oriented

| Name | Description |
| ---- | ----------- |
| [Artificial Analysis](https://artificialanalysis.ai) | Artificial Analysis is a platform to help users make informed decisions on AI model selection and hosting providers. |
| [Papers Leaderboard](https://huggingface.co/spaces/ameerazam08/Paper-LeaderBoard) | Papers Leaderboard is a platform to evaluate the popularity of machine learning papers. |
| [Provider Leaderboard](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena) | LLM API Providers Leaderboard is a platform to compare API provider performance for over LLM endpoints across performance key metrics. |

## Data-oriented

| Name | Description |
| ---- | ----------- |
| [DataComp - CLIP](https://www.datacomp.ai/dcclip/leaderboard.html) | DataComp - CLIP is a benchmark to evaluate the performance of various image/text pairs when used with a fixed model architecture. |
| [DataComp - LM](https://www.datacomp.ai/dclm/leaderboard.html) | DataComp - CLIP is a benchmark to evaluate the performance of various text datasets when used with a fixed model architecture. |

## Metric-oriented

| Name | Description |
| ---- | ----------- |
| [AlignScore](https://github.com/yuh-zha/AlignScore) | AlignScore evaluates the performance of different metrics in assessing factual consistency. |

## Meta Leaderboard

| Name | Description |
| ---- | ----------- |
| [Open Leaderboards Leaderboard](https://huggingface.co/spaces/mrfakename/open-leaderboards-leaderboard) | Open Leaderboards Leaderboard is a meta-leaderboard that leverages human preferences to compare machine learning leaderboards. |
