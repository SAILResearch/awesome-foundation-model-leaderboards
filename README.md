<div align="center">
    <img src="https://github.com/user-attachments/assets/7d987c01-a398-413c-9860-c1704352a7fe" alt="lbops" width="300"/>
    <h1>Awesome Foundation Model Leaderboard</h1>
    <a href="https://awesome.re">
        <img src="https://awesome.re/badge.svg" height="20"/>
    </a>
    <a href="https://github.com/SAILResearch/awesome-foundation-model-leaderboards/fork">
        <img src="https://img.shields.io/badge/PRs-Welcome-red" height="20"/>
    </a>
    <a href="https://arxiv.org/pdf/2407.04065.pdf">
        <img src="https://img.shields.io/badge/Arxiv-2407.04065-red" height="20"/>
    </a>
</div>

**Awesome Foundation Model Leaderboard** is a curated list of awesome foundation model leaderboards (for an explanation of what a leaderboard is, please refer to this [tutorial](https://huggingface.co/docs/leaderboards/index)), along with various development tools and evaluation organizations according to [our survey](https://arxiv.org/abs/2407.04065):

<p align="center"><strong>On the Workflows and Smells of Leaderboard Operations (LBOps):<br>An Exploratory Study of Foundation Model Leaderboards</strong></p>

<p align="center"><a href="https://github.com/zhimin-z">Zhimin (Jimmy) Zhao</a>, <a href="https://abdulali.github.io">Abdul Ali Bangash</a>, <a href="https://www.filipecogo.pro">Filipe Roseiro Côgo</a>, <a href="https://mcis.cs.queensu.ca/bram.html">Bram Adams</a>, <a href="https://research.cs.queensu.ca/home/ahmed">Ahmed E. Hassan</a></p>

<p align="center"><a href="https://sail.cs.queensu.ca">Software Analysis and Intelligence Lab (SAIL)</a></p>

If you find this repository useful, please consider giving us a star :star: and citation:

```
@article{zhao2025workflows,
  title={On the Workflows and Smells of Leaderboard Operations (LBOps): An Exploratory Study of Foundation Model Leaderboards},
  author={Zhao, Zhimin and Bangash, Abdul Ali and C{\^o}go, Filipe Roseiro and Adams, Bram and Hassan, Ahmed E},
  journal={IEEE Transactions on Software Engineering},
  year={2025},
  publisher={IEEE}
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
- [**Challenges**](#challenges)
- [**Rankings**](#rankings)
    - [Model Ranking](#model-ranking)
        - [Comprehensive](#comprehensive)
        - [Text](#text)
        - [Code](#code)
        - [Image](#image)
        - [Video](#video)
        - [Math](#math)
        - [Agent](#agent)
        - [Medical](#medical)
        - [Audio](#audio)
        - [3D](#3d)
        - [Game](#game)
        - [Multimodal](#multimodal)
        - [Intelligence Quotient](#intelligence-quotient)
    - [Database Ranking](#database-ranking)
    - [Dataset Ranking](#dataset-ranking)
    - [Metric Ranking](#metric-ranking)
    - [Paper Ranking](#paper-ranking)
    - [Leaderboard Ranking](#leaderboard-ranking)



# Tools

| Name | Description |
| ---- | ----------- |
| [Demo Leaderboard](https://huggingface.co/spaces/demo-leaderboard-backend/leaderboard) | Demo leaderboard helps users easily deploy their leaderboards with a standardized template. |
| [Demo Leaderboard Backend](https://huggingface.co/spaces/demo-leaderboard-backend/backend) | Demo leaderboard backend helps users manage the leaderboard and handle submission requests, check [this](https://huggingface.co/docs/leaderboards/leaderboards/building_page) for details. |
| [Kaggle Competition Creation](https://www.kaggle.com/competitions?new=true) | Kaggle Competition Creation enables you to design and launch custom competitions, leveraging your datasets to engage the data science community. |
| [Leaderboard Explorer](https://huggingface.co/spaces/leaderboards/LeaderboardsExplorer) | Leaderboard Explorer helps users navigate the diverse range of leaderboards available on Hugging Face Spaces. |
| [Open LLM Leaderboard Renamer](https://huggingface.co/spaces/Weyaxi/open-llm-leaderboard-renamer) | open-llm-leaderboard-renamer helps users rename their models in Open LLM Leaderboard easily. |
| [Open LLM Leaderboard Results PR Opener](https://huggingface.co/spaces/Weyaxi/leaderboard-results-to-modelcard) | Open LLM Leaderboard Results PR Opener helps users showcase Open LLM Leaderboard results in their model cards. |
| [Open LLM Leaderboard Scraper](https://github.com/Weyaxi/scrape-open-llm-leaderboard) | Open LLM Leaderboard Scraper helps users scrape and export data from Open LLM Leaderboard. |



# Challenges

| Name | Description |
| ---- | ----------- |
| [AIcrowd](https://www.aicrowd.com/challenges) | AIcrowd hosts machine learning challenges and competitions across domains such as computer vision, NLP, and reinforcement learning, aimed at both researchers and practitioners. |
| [AI Hub](https://eu.aihub.ml/competitions) | AI Hub offers a variety of competitions to encourage AI solutions to real-world problems, with a focus on innovation and collaboration. |
| [AI Studio](https://aistudio.baidu.com/competition) | AI Studio offers AI competitions mainly for computer vision, NLP, and other data-driven tasks, allowing users to develop and showcase their AI skills. |
| [Allen Institute for AI](https://leaderboard.allenai.org) | The Allen Institute for AI provides leaderboards and benchmarks on tasks in natural language understanding, commonsense reasoning, and other areas in AI research. |
| [Codabench](https://www.codabench.org) | Codabench is an open-source platform for benchmarking AI models, enabling customizable, user-driven challenges across various AI domains. |
| [DataFountain](https://www.datafountain.cn/competitions) | DataFountain is a Chinese AI competition platform featuring challenges in finance, healthcare, and smart cities, encouraging solutions for industry-related problems. |
| [DrivenData](https://www.drivendata.org/competitions) | DrivenData hosts machine learning challenges with a social impact, aiming to solve issues in areas, such as public health, disaster relief, and sustainable development. |
| [Dynabench](https://dynabench.org/tasks) | Dynabench offers dynamic benchmarks where models are evaluated continuously, often involving human interaction, to ensure robustness in evolving AI tasks. |
| [Eval AI](https://eval.ai/web/challenges/list) | EvalAI is a platform for hosting and participating in AI challenges, widely used by researchers for benchmarking models in tasks, such as image classification, NLP, and reinforcement learning. |
| [Grand Challenge](https://grand-challenge.org/challenges) | Grand Challenge provides a platform for medical imaging challenges, supporting advancements in medical AI, particularly in areas, such as radiology and pathology. |
| [Hilti](https://www.hilti-challenge.com) | Hilti hosts challenges aimed at advancing AI and machine learning in the construction industry, with a focus on practical, industry-relevant applications. |
| [InsightFace](https://insightface.ai/challenges) | InsightFace focuses on AI challenges related to face recognition, verification, and analysis, supporting advancements in identity verification and security. |
| [Kaggle](https://www.kaggle.com/competitions) | Kaggle is one of the largest platforms for data science and machine learning competitions, covering a broad range of topics from image classification to NLP and predictive modeling. |
| [nuScenes](https://www.nuscenes.org) | nuScenes enables researchers to study challenging urban driving situations using the full sensor suite of a real self-driving car, facilitating research in autonomous driving. |
| [Robust Reading Competition](https://rrc.cvc.uab.es) | Robust Reading refers to the research area on interpreting written communication in unconstrained settings, with competitions focused on text recognition in real-world environments. |
| [Tianchi](https://tianchi.aliyun.com/competition) | Tianchi, hosted by Alibaba, offers a range of AI competitions, particularly popular in Asia, with a focus on commerce, healthcare, and logistics. |



# Rankings

## Model Ranking

### Comprehensive

| Name | Description |
| ---- | ----------- |
| [AI Benchmarking Hub](https://epoch.ai/benchmarks) | AI Benchmarking Hub tracks and compares AI model performance in reasoning, coding, and knowledge tasks. |
| [Artificial Analysis](https://artificialanalysis.ai) | Artificial Analysis is a platform to help users make informed decisions on AI model selection and hosting providers. |
| [CompassRank](https://rank.opencompass.org.cn) | CompassRank is a platform to offer a comprehensive, objective, and neutral evaluation reference of foundation mdoels for the industry and research. |
| [FlagEval](https://flageval.baai.ac.cn/#/leaderboard) | FlagEval is a comprehensive platform for evaluating foundation models. |
| [Generative AI Leaderboards](https://accubits.com/generative-ai-models-leaderboard) | Generative AI Leaderboard ranks the top-performing generative AI models based on various metrics. |
| [Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm) | Holistic Evaluation of Language Models (HELM) is a reproducible and transparent framework for evaluating foundation models. |
| [LMArena](https://lmarena.ai/leaderboard) | LMArena operates a chatbot arena where various foundation models compete based on user preferences across multiple categories: text generation, web development, computer vision, text-to-image synthesis, search capabilities, and coding assistance. |
| [Openrouter Leaderboard](https://openrouter.ai/rankings) | Openrouter Leaderboard offers a real-time comparison of language models based on normalized token usage for prompts and completions, updated frequently. |
| [Papers With Code](https://paperswithcode.com) | Papers With Code provides open-source leaderboards and benchmarks, linking AI research papers with code to foster transparency and reproducibility in machine learning. |
| [SuperCLUE](https://www.superclueai.com) | SuperCLUE is a series of benchmarks for evaluating Chinese foundation models. |
| [Vals AI](https://www.vals.ai) | Val AI builds custom, industry-specific benchmarks using private datasets to provide unbiased third-party evaluations of LLM performance. |
| [Vellum LLM Leaderboard](https://www.vellum.ai/llm-leaderboard) | Vellum LLM Leaderboard shows a comparison of capabilities, price and context window for leading commercial and open-source LLMs. |
| [Yupp Leaderboard](https://yupp.ai/leaderboard) | Yupp is a platform that enables users to compare outputs from multiple AI models side by side, select their preferred response, and provide feedback. |

### Text

| Name | Description |
| ---- | ----------- |
| [ACLUE](https://github.com/isen-zhang/ACLUE/blob/main/README_EN.md#leaderboard-) | ACLUE is an evaluation benchmark for ancient Chinese language comprehension. |
| [African Languages LLM Eval Leaderboard](https://huggingface.co/spaces/taresco/open_african_languages_eval_leaderboard) | African Languages LLM Eval Leaderboard tracks progress and ranks performance of LLMs on African languages. |
| [AGIEval](https://github.com/ruixiangcui/AGIEval?tab=readme-ov-file#leaderboard) | AGIEval is a human-centric benchmark to evaluate the general abilities of foundation models in tasks pertinent to human cognition and problem-solving. |
| [Aiera Leaderboard](https://huggingface.co/spaces/Aiera/aiera-finance-leaderboard) | Aiera Leaderboard evaluates LLM performance on financial intelligence tasks, including speaker assignments, speaker change identification, abstractive summarizations, calculation-based Q&A, and financial sentiment tagging. |
| [AIR-Bench](https://huggingface.co/spaces/AIR-Bench/leaderboard) | AIR-Bench is a benchmark to evaluate heterogeneous information retrieval capabilities of language models. |
| [AI Energy Score Leaderboard](https://huggingface.co/spaces/EnergyStarAI/2024_Leaderboard) | AI Energy Score Leaderboard tracks and compares different models in energy efficiency. |
| [ai-benchmarks](https://github.com/fixie-ai/ai-benchmarks?tab=readme-ov-file#leaderboard) | ai-benchmarks contains a handful of evaluation results for the response latency of popular AI services. |
| [AlignBench](https://llmbench.ai/align/data) | AlignBench is a multi-dimensional benchmark for evaluating LLMs' alignment in Chinese. |
| [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval) | AlpacaEval is an automatic evaluator designed for instruction-following LLMs. |
| [ANGO](https://huggingface.co/spaces/AngoHF/ANGO-Leaderboard) | ANGO is a generation-oriented Chinese language model evaluation benchmark. |
| [Arabic Tokenizers Leaderboard](https://huggingface.co/spaces/MohamedRashad/arabic-tokenizers-leaderboard) | Arabic Tokenizers Leaderboard compares the efficiency of LLMs in parsing Arabic in its different dialects and forms. |
| [Arena-Hard-Auto](https://github.com/lmarena/arena-hard-auto?tab=readme-ov-file#full-leaderboard-updated-0831) | Arena-Hard-Auto is a benchmark for instruction-tuned LLMs. |
| [AutoRace](https://www.llm-reasoners.net/leaderboard) | AutoRace focuses on the direct evaluation of LLM reasoning chains with metric AutoRace (Automated Reasoning Chain Evaluation). |
| [Auto Arena](https://huggingface.co/spaces/Auto-Arena/Leaderboard) | Auto Arena is a benchmark in which various language model agents engage in peer-battles to evaluate their performance. |
| [Auto-J](https://github.com/GAIR-NLP/auto-j?tab=readme-ov-file#leaderboard) | Auto-J hosts evaluation results on the pairwise response comparison and critique generation tasks. |
| [BABILong](https://huggingface.co/spaces/RMT-team/babilong) | BABILong is a benchmark for evaluating the performance of language models in processing arbitrarily long documents with distributed facts. |
| [BBL](https://github.com/google/BIG-bench#big-bench-lite-leaderboard) | BBL (BIG-bench Lite) is a small subset of 24 diverse JSON tasks from BIG-bench. It is designed to provide a canonical measure of model performance, while being far cheaper to evaluate than the full set of more than 200 programmatic and JSON tasks in BIG-bench. |
| [BeHonest](https://gair-nlp.github.io/BeHonest/#leaderboard) | BeHonest is a benchmark to evaluate honesty - awareness of knowledge boundaries (self-knowledge), avoidance of deceit (non-deceptiveness), and consistency in responses (consistency) - in LLMs. |
| [BenBench](https://gair-nlp.github.io/benbench) | BenBench is a benchmark to evaluate the extent to which LLMs conduct verbatim training on the training set of a benchmark over the test set to enhance capabilities. |
| [BenCzechMark](https://huggingface.co/spaces/CZLC/BenCzechMark) | BenCzechMark (BCM) is a multitask and multimetric Czech language benchmark for LLMs with a unique scoring system that utilizes the theory of statistical significance. |
| [BiGGen-Bench](https://huggingface.co/spaces/prometheus-eval/BiGGen-Bench-Leaderboard) | BiGGen-Bench is a comprehensive benchmark to evaluate LLMs across a wide variety of tasks. |
| [BotChat](https://botchat.opencompass.org.cn) | BotChat is a benchmark to evaluate the multi-round chatting capabilities of LLMs through a proxy task. |
| [CaselawQA](https://huggingface.co/spaces/ricdomolm/caselawqa_leaderboard) | CaselawQA is a benchmark comprising legal classification tasks derived from the Supreme Court and Songer Court of Appeals legal databases. |
| [CFLUE](https://github.com/aliyun/cflue?tab=readme-ov-file#leaderboard) | CFLUE is a benchmark to evaluate LLMs' understanding and processing capabilities in the Chinese financial domain. |
| [Ch3Ef](https://openlamm.github.io/ch3ef/leaderboard.html) | Ch3Ef is a benchmark to evaluate alignment with human expectations using 1002 human-annotated samples across 12 domains and 46 tasks based on the hhh principle. |
| [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub?tab=readme-ov-file#leaderboard---main) | Chain-of-Thought Hub is a benchmark to evaluate the reasoning capabilities of LLMs. |
| [ChemBench](https://lamalab-org.github.io/chem-bench/leaderboard) | ChemBench is a benchmark to evaluate the chemical knowledge and reasoning abilities of LLMs. |
| [Chinese SimpleQA](http://47.109.32.164) | Chinese SimpleQA is a Chinese benchmark to evaluate the factuality ability of language models to answer short questions. |
| [CLEM Leaderboard](https://huggingface.co/spaces/colab-potsdam/clem-leaderboard) | CLEM is a framework designed for the systematic evaluation of chat-optimized LLMs as conversational agents. |
| [CLEVA](http://www.lavicleva.com) | CLEVA is a benchmark to evaluate LLMs on 31 tasks using 370K Chinese queries from 84 diverse datasets and 9 metrics. |
| [Chinese Large Model Leaderboard](https://github.com/jeinlee1991/chinese-llm-benchmark?tab=readme-ov-file#-%E6%8E%92%E8%A1%8C%E6%A6%9C) | Chinese Large Model Leaderboard is a platform to evaluate the performance of Chinese LLMs. |
| [CMMLU](https://github.com/haonan-li/CMMLU/blob/master/README_EN.md#leaderboard) | CMMLU is a benchmark to evaluate the performance of LLMs in various subjects within the Chinese cultural context. |
| [CMMMU](https://cmmmu-benchmark.github.io/#leaderboard) | CMMMU is a benchmark to evaluate LMMs on tasks demanding college-level subject knowledge and deliberate reasoning in a Chinese context. |
| [CommonGen](https://inklab.usc.edu/CommonGen/leaderboard.html) | CommonGen is a benchmark to evaluate generative commonsense reasoning by testing machines on their ability to compose coherent sentences using a given set of common concepts. |
| [CompMix](https://qa.mpi-inf.mpg.de/compmix) | CompMix is a benchmark for heterogeneous question answering. |
| [Compression Rate Leaderboard](https://huggingface.co/spaces/xu-song/tokenizer-arena) | Compression Rate Leaderboard aims to evaluate tokenizer performance on different languages. |
| [Compression Leaderboard](https://huggingface.co/spaces/eson/tokenizer-arena) | Compression Leaderboard is a platform to evaluate the compression performance of LLMs. |
| [CopyBench](https://huggingface.co/spaces/chentong00/CopyBench-leaderboard) | CopyBench is a benchmark to evaluate the copying behavior and utility of language models as well as the effectiveness of methods to mitigate copyright risks. |
| [CoTaEval](https://huggingface.co/spaces/boyiwei/CoTaEval_leaderboard) | CoTaEval is a benchmark to evaluate the feasibility and side effects of copyright takedown methods for LLMs. |
| [ConvRe](https://huggingface.co/spaces/3B-Group/ConvRe-Leaderboard) | ConvRe is a benchmark to evaluate LLMs' ability to comprehend converse relations. |
| [CriticEval](https://open-compass.github.io/CriticEval) | CriticEval is a benchmark to evaluate LLMs' ability to make critique responses. |
| [CS-Bench](https://csbench.github.io/#leaderboard) | CS-Bench is a bilingual benchmark designed to evaluate LLMs' performance across 26 computer science subfields, focusing on knowledge and reasoning. |
| [CUTE](https://huggingface.co/spaces/leukas/cute_leaderboard) | CUTE is a benchmark to test the orthographic knowledge of LLMs. |
| [CyberMetric](https://github.com/cybermetric/CyberMetric?tab=readme-ov-file#llm-leaderboard-on-cybermetric-dataset) | CyberMetric is a benchmark to evaluate the cybersecurity knowledge of LLMs. |
| [CzechBench](https://huggingface.co/spaces/CIIRC-NLP/czechbench_leaderboard) | CzechBench is a benchmark to evaluate Czech language models. |
| [C-Eval](https://cevalbenchmark.com/static/leaderboard.html) | C-Eval is a Chinese evaluation suite for LLMs. |
| [Decentralized Arena Leaderboard](https://huggingface.co/spaces/LLM360/de-arena) | Decentralized Arena hosts a decentralized and democratic platform for LLM evaluation, automating and scaling assessments across diverse, user-defined dimensions, including mathematics, logic, and science. |
| [DecodingTrust](https://decodingtrust.github.io/leaderboard) | DecodingTrust is a platform to evaluate the trustworthiness of LLMs. |
| [Domain LLM Leaderboard](https://huggingface.co/spaces/NexaAIDev/domain_llm_leaderboard) | Domain LLM Leaderboard is a platform to evaluate the popularity of domain-specific LLMs. |
| [Enterprise Scenarios leaderboard](https://huggingface.co/spaces/PatronusAI/enterprise_scenarios_leaderboard) | Enterprise Scenarios Leaderboard tracks and evaluates the performance of LLMs on real-world enterprise use cases. |
| [EQ-Bench](https://eqbench.com) | EQ-Bench is a benchmark to evaluate aspects of emotional intelligence in LLMs. |
| [European LLM Leaderboard](https://huggingface.co/spaces/openGPT-X/european-llm-leaderboard) | European LLM Leaderboard tracks and compares performance of LLMs in European languages. |
| [EvalGPT.ai](https://github.com/h2oai/h2o-LLM-eval?tab=readme-ov-file#elo-leaderboard) | EvalGPT.ai hosts a chatbot arena to compare and rank the performance of LLMs. |
| [Eval Arena](https://crux-eval.github.io/eval-arena) | Eval Arena measures noise levels, model quality, and benchmark quality by comparing model pairs across several LLM evaluation benchmarks with example-level analysis and pairwise comparisons. |
| [Factuality Leaderboard](https://github.com/gair-nlp/factool?tab=readme-ov-file#factuality-leaderboard) | Factuality Leaderboard compares the factual capabilities of LLMs. |
| [FanOutQA](https://fanoutqa.com/leaderboard) | FanOutQA is a high quality, multi-hop, multi-document benchmark for LLMs using English Wikipedia as its knowledge base. |
| [FastEval](https://fasteval.github.io/FastEval) | FastEval is a toolkit for quickly evaluating instruction-following and chat language models on various benchmarks with fast inference and detailed performance insights. |
| [FELM](https://hkust-nlp.github.io/felm) | FELM is a meta benchmark to evaluate factuality evaluation benchmark for LLMs. |
| [FinEval](https://github.com/SUFE-AIFLM-Lab/FinEval?tab=readme-ov-file#performance-leaderboard) | FinEval is a benchmark to evaluate financial domain knowledge in LLMs. |
| [Fine-tuning Leaderboard](https://predibase.com/fine-tuning-index) | Fine-tuning Leaderboard is a platform to rank and showcase models that have been fine-tuned using open-source datasets or frameworks. |
| [Flames](https://github.com/AIFlames/Flames?tab=readme-ov-file#-leaderboard) | Flames is a highly adversarial Chinese benchmark for evaluating LLMs' value alignment across fairness, safety, morality, legality, and data protection. |
| [FollowBench](https://github.com/YJiangcm/FollowBench?tab=readme-ov-file#%EF%B8%8F-leaderboard) | FollowBench is a multi-level fine-grained constraints following benchmark to evaluate the instruction-following capability of LLMs. |
| [Forbidden Question Dataset](https://junjie-chu.github.io/Public_Comprehensive_Assessment_Jailbreak/leaderboard) | Forbidden Question Dataset is a benchmark containing 160 questions from 160 violated categories, with corresponding targets for evaluating jailbreak methods. |
| [FuseReviews](https://huggingface.co/spaces/lovodkin93/FuseReviews-Leaderboard) | FuseReviews aims to advance grounded text generation tasks, including long-form question-answering and summarization. |
| [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) | GAIA aims to test fundamental abilities that an AI assistant should possess. |
| [GAVIE](https://github.com/FuxiaoLiu/LRV-Instruction?tab=readme-ov-file#leaderboards) | GAVIE is a GPT-4-assisted benchmark for evaluating hallucination in LMMs by scoring accuracy and relevancy without relying on human-annotated groundtruth. |
| [GPT-Fathom](https://github.com/GPT-Fathom/GPT-Fathom?tab=readme-ov-file#-leaderboard) | GPT-Fathom is an LLM evaluation suite, benchmarking 10+ leading LLMs as well as OpenAI's legacy models on 20+ curated benchmarks across 7 capability categories, all under aligned settings. |
| [GrailQA](https://dki-lab.github.io/GrailQA) | Strongly Generalizable Question Answering (GrailQA) is a large-scale, high-quality benchmark for question answering on knowledge bases (KBQA) on Freebase with 64,331 questions annotated with both answers and corresponding logical forms in different syntax (i.e., SPARQL, S-expression, etc.). |
| [Guerra LLM AI Leaderboard](https://huggingface.co/spaces/luisrguerra/guerra-llm-ai-leaderboard) | Guerra LLM AI Leaderboard compares and ranks the performance of LLMs across quality, price, performance, context window, and others. |
| [Hallucinations Leaderboard](https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard) | Hallucinations Leaderboard aims to track, rank and evaluate hallucinations in LLMs. |
| [HalluQA](https://github.com/OpenMOSS/HalluQA?tab=readme-ov-file#leaderboard) | HalluQA is a benchmark to evaluate the phenomenon of hallucinations in Chinese LLMs. |
| [Hebrew LLM Leaderboard](https://huggingface.co/spaces/hebrew-llm-leaderboard/leaderboard) | Hebrew LLM Leaderboard tracks and ranks language models according to their success on various tasks on Hebrew. |
| [HellaSwag](https://rowanzellers.com/hellaswag) | HellaSwag is a benchmark to evaluate common-sense reasoning in LLMs. |
| [Hughes Hallucination Evaluation Model leaderboard](https://huggingface.co/spaces/vectara/leaderboard) | Hughes Hallucination Evaluation Model leaderboard is a platform to evaluate how often a language model introduces hallucinations when summarizing a document. |
| [Icelandic LLM leaderboard](https://huggingface.co/spaces/mideind/icelandic-llm-leaderboard) | Icelandic LLM leaderboard tracks and compare models on Icelandic-language tasks. |
| [IFEval](https://huggingface.co/spaces/Krisseck/IFEval-Leaderboard) | IFEval is a benchmark to evaluate LLMs' instruction following capabilities with verifiable instructions. |
| [IL-TUR](https://huggingface.co/spaces/Exploration-Lab/IL-TUR-Leaderboard) | IL-TUR is a benchmark for evaluating language models on monolingual and multilingual tasks focused on understanding and reasoning over Indian legal documents. |
| [Indic LLM Leaderboard](https://huggingface.co/spaces/Cognitive-Lab/indic_llm_leaderboard) | Indic LLM Leaderboard is platform to track and compare the performance of Indic LLMs. |
| [Indico LLM Leaderboard](https://indicodata.ai/llm) | Indico LLM Leaderboard evaluates and compares the accuracy of various language models across providers, datasets, and capabilities like text classification, key information extraction, and generative summarization. |
| [InstructEval](https://declare-lab.github.io/instruct-eval) | InstructEval is a suite to evaluate instruction selection methods in the context of LLMs. |
| [Italian LLM-Leaderboard](https://huggingface.co/spaces/rstless-research/italian_open_llm_leaderboard) | Italian LLM-Leaderboard tracks and compares LLMs in Italian-language tasks. |
| [JailbreakBench](https://jailbreakbench.github.io) | JailbreakBench is a benchmark for evaluating LLM vulnerabilities through adversarial prompts. |
| [Japanese Chatbot Arena](https://huggingface.co/spaces/yutohub/japanese-chatbot-arena-leaderboard) | Japanese Chatbot Arena hosts the chatbot arena, where various LLMs compete based on their performance in Japanese. |
| [Japanese Language Model Financial Evaluation Harness](https://github.com/pfnet-research/japanese-lm-fin-harness?tab=readme-ov-file#0-shot-leaderboard) | Japanese Language Model Financial Evaluation Harness is a harness for Japanese language model evaluation in the financial domain. |
| [Japanese LLM Roleplay Benchmark](https://github.com/oshizo/japanese-llm-roleplay-benchmark?tab=readme-ov-file#leaderboard-v20231103) | Japanese LLM Roleplay Benchmark is a benchmark to evaluate the performance of Japanese LLMs in character roleplay. |
| [JMMMU](https://huggingface.co/spaces/JMMMU/JMMMU_Leaderboard) | JMMMU (Japanese MMMU) is a multimodal benchmark to evaluate LMM performance in Japanese. |
| [JustEval](https://allenai.github.io/re-align/just_eval.html) | JustEval is a powerful tool designed for fine-grained evaluation of LLMs. |
| [KoLA](http://103.238.162.37:31622/LeaderBoard) | KoLA is a benchmark to evaluate the world knowledge of LLMs. |
| [LaMP](https://lamp-benchmark.github.io/leaderboard) | LaMP (Language Models Personalization) is a benchmark to evaluate personalization capabilities of language models. |
| [Language Model Council](https://llm-council.com) | Language Model Council (LMC) is a benchmark to evaluate tasks that are highly subjective and often lack majoritarian human agreement. |
| [LawBench](https://lawbench.opencompass.org.cn/leaderboard) | LawBench is a benchmark to evaluate the legal capabilities of LLMs. |
| [La Leaderboard](https://huggingface.co/spaces/la-leaderboard/la-leaderboard) | La Leaderboard evaluates and tracks LLM memorization, reasoning and linguistic capabilities in Spain, LATAM and Caribbean. |
| [LogicKor](https://lk.instruct.kr) | LogicKor is a benchmark to evaluate the multidisciplinary thinking capabilities of Korean LLMs. |
| [LongICL Leaderboard](https://huggingface.co/spaces/TIGER-Lab/LongICL-Leaderboard) | LongICL Leaderboard is a platform to evaluate long in-context learning evaluations for LLMs. |
| [LooGLE](https://bigai-nlco.github.io/LooGLE/#-capability-leaderboard) | LooGLE is a benchmark to evaluate long context understanding capabilties of LLMs. |
| [LAiW](https://huggingface.co/spaces/daishen/SCULAiW) | LAiW is a benchmark to evaluate Chinese legal language understanding and reasoning. |
| [LLM Benchmarker Suite](https://llm-evals.formula-labs.com) | LLM Benchmarker Suite is a benchmark to evaluate the comprehensive capabilities of LLMs. |
| [Large Language Model Assessment in English Contexts](https://huggingface.co/spaces/CathieDaDa/LLM_leaderboard_en) | Large Language Model Assessment in English Contexts is a platform to evaluate LLMs in the English context. |
| [Large Language Model Assessment in the Chinese Context](https://huggingface.co/spaces/CathieDaDa/LLM_leaderboard) | Large Language Model Assessment in the Chinese Context is a platform to evaluate LLMs in the Chinese context. |
| [LegalBench](https://www.legalevalhub.ai/leaderboard/legalbench_full) | LegalBench is a comprehensive benchmark for evaluating legal reasoning in LLMs. |
| [LIBRA](https://huggingface.co/spaces/ai-forever/LIBRA-Leaderboard) | LIBRA is a benchmark for evaluating LLMs' capabilities in understanding and processing long Russian text. |
| [LibrAI-Eval GenAI Leaderboard](https://test.leaderboard.librai.tech/LeaderBoard) | LibrAI-Eval GenAI Leaderboard focuses on the balance between the LLM’s capability and safety in English. |
| [LiveBench](https://livebench.ai) | LiveBench is a benchmark for LLMs to minimize test set contamination and enable objective, automated evaluation across diverse, regularly updated tasks. |
| [LLMEval](http://llmeval.com) | LLMEval is a benchmark to evaluate the quality of open-domain conversations with LLMs. |
| [Llmeval-Gaokao2024-Math](https://github.com/llmeval/Llmeval-Gaokao2024-Math?tab=readme-ov-file#%E8%AF%84%E6%B5%8B%E7%BB%93%E6%9E%9C) | Llmeval-Gaokao2024-Math is a benchmark for evaluating LLMs on 2024 Gaokao-level math problems in Chinese. | 
| [LLMHallucination Leaderboard](https://huggingface.co/spaces/ramiroluo/LLMHallucination_Leaderboard) | Hallucinations Leaderboard evaluates LLMs based on an array of hallucination-related benchmarks. |
| [LLMPerf](https://github.com/ray-project/llmperf-leaderboard) | LLMPerf is a tool to evaluate the performance of LLMs using both load and correctness tests. |
| [LLMs Disease Risk Prediction Leaderboard](https://huggingface.co/spaces/TemryL/LLM-Disease-Risk-Leaderboard) | LLMs Disease Risk Prediction Leaderboard is a platform to evaluate LLMs on disease risk prediction. |
| [LLM Leaderboard](https://klu.ai/llm-leaderboard) | LLM Leaderboard tracks and evaluates LLM providers, enabling selection of the optimal API and model for user needs. |
| [LLM Leaderboard for CRM](https://huggingface.co/spaces/Salesforce/crm_llm_leaderboard) | CRM LLM Leaderboard is a platform to evaluate the efficacy of LLMs for business applications. |
| [LLM Observatory](https://ai-sandbox.list.lu/llm-leaderboard) | LLM Observatory is a benchmark that assesses and ranks LLMs based on their performance in avoiding social biases across categories like LGBTIQ+ orientation, age, gender, politics, race, religion, and xenophobia. |
| [LLM Price Leaderboard](https://huggingface.co/spaces/seawolf2357/leaderboard_llm_price) | LLM Price Leaderboard tracks and compares LLM costs based on one million tokens. |
| [LLM Safety Leaderboard](https://huggingface.co/spaces/AI-Secure/llm-trustworthy-leaderboard) | LLM Safety Leaderboard aims to provide a unified evaluation for language model safety. |
| [LLM Use Case Leaderboard](https://llmleaderboard.goml.io) | LLM Use Case Leaderboard tracks and evaluates LLMs in business usecases. |
| [LLM-AggreFact](https://llm-aggrefact.github.io) | LLM-AggreFact is a fact-checking benchmark that aggregates most up-to-date publicly available datasets on grounded factuality evaluation. |
| [LLM-Leaderboard](https://github.com/LudwigStumpp/llm-leaderboard?tab=readme-ov-file#leaderboard) | LLM-Leaderboard is a joint community effort to create one central leaderboard for LLMs. |
| [LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard) | LLM-Perf Leaderboard aims to benchmark the performance of LLMs with different hardware, backends, and optimizations. |
| [LMExamQA](https://lmexam.com) | LMExamQA is a benchmarking framework where a language model acts as an examiner to generate questions and evaluate responses in a reference-free, automated manner for comprehensive, equitable assessment. |
| [LongBench](https://github.com/THUDM/LongBench?tab=readme-ov-file#%EF%B8%8F-leaderboard) | LongBench is a benchmark for assessing the long context understanding capabilities of LLMs. |
| [Loong](https://github.com/MozerWang/Loong?tab=readme-ov-file#leaderboard) | Loong is a long-context benchmark for evaluating LLMs' multi-document QA abilities across financial, legal, and academic scenarios. |
| [Low-bit Quantized Open LLM Leaderboard](https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard) | Low-bit Quantized Open LLM Leaderboard tracks and compares quantization LLMs with different quantization algorithms. |
| [LV-Eval](https://github.com/infinigence/LVEval?tab=readme-ov-file#leaderboard) | LV-Eval is a long-context benchmark with five length levels and advanced techniques for accurate evaluation of LLMs on single-hop and multi-hop QA tasks across bilingual datasets. |
| [LucyEval](http://lucyeval.besteasy.com/leaderboard.html) | LucyEval offers a thorough assessment of LLMs' performance in various Chinese contexts. |
| [L-Eval](https://l-eval.github.io) | L-Eval is a Long Context Language Model (LCLM) evaluation benchmark to evaluate the performance of handling extensive context. |
| [M3KE](https://github.com/tjunlp-lab/M3KE?tab=readme-ov-file#evaluation-leaderboard-more-models-to-be-added) | M3KE is a massive multi-level multi-subject knowledge evaluation benchmark to measure the knowledge acquired by Chinese LLMs. |
| [MetaCritique](https://github.com/GAIR-NLP/MetaCritique?tab=readme-ov-file#leaderboard) | MetaCritique is a judge that can evaluate human-written or LLMs-generated critique by generating critique. |
| [MINT](https://xwang.dev/mint-bench) | MINT is a benchmark to evaluate LLMs' ability to solve tasks with multi-turn interactions by using tools and leveraging natural language feedback. |
| [Meta Open LLM leaderboard](https://huggingface.co/spaces/felixz/meta_open_llm_leaderboard) | The Meta Open LLM leaderboard serves as a central hub for consolidating data from various open LLM leaderboards into a single, user-friendly visualization page. |
| [MIMIC Clinical Decision Making Leaderboard](https://huggingface.co/spaces/MIMIC-CDM/leaderboard) | MIMIC Clinical Decision Making Leaderboard tracks and evaluates LLms in realistic clinical decision-making for abdominal pathologies. |
| [MixEval](https://mixeval.github.io/#leaderboard) | MixEval is a benchmark to evaluate LLMs via by strategically mixing off-the-shelf benchmarks. |
| [ML.ENERGY Leaderboard](https://ml.energy/leaderboard) | ML.ENERGY Leaderboard evaluates the energy consumption of LLMs. |
| [MMLU](https://github.com/hendrycks/test?tab=readme-ov-file#test-leaderboard) | MMLU is a benchmark to evaluate the performance of LLMs across a wide array of natural language understanding tasks. |
| [MMLU-by-task Leaderboard](https://huggingface.co/spaces/CoreyMorris/MMLU-by-task-Leaderboard) | MMLU-by-task Leaderboard provides a platform for evaluating and comparing various ML models across different language understanding tasks. |
| [MMLU-Pro](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro) | MMLU-Pro is a more challenging version of MMLU to evaluate the reasoning capabilities of LLMs. |
| [ModelScope LLM Leaderboard](https://modelscope.cn/leaderboard/58/ranking?type=free) | ModelScope LLM Leaderboard is a platform to evaluate LLMs objectively and comprehensively. |
| [Model Evaluation Leaderboard](https://github.com/databricks/databricks-ml-examples?tab=readme-ov-file#model-evaluation-leaderboard) | Model Evaluation Leaderboard tracks and evaluates text generation models based on their performance across various benchmarks using Mosaic Eval Gauntlet framework. |
| [MSNP Leaderboard](https://huggingface.co/spaces/evilfreelancer/msnp-leaderboard) | MSNP Leaderboard tracks and evaluates quantized GGUF models' performance on various GPU and CPU combinations using single-node setups via Ollama. |
| [MSTEB](https://huggingface.co/spaces/clibrain/Spanish-Embeddings-Leaderboard) | MSTEB is a benchmark for measuring the performance of text embedding models in Spanish. |
| [MTEB](https://huggingface.co/spaces/mteb/leaderboard) | MTEB is a massive benchmark for measuring the performance of text embedding models on diverse embedding tasks across 112 languages. |
| [MTEB Arena](https://huggingface.co/spaces/mteb/arena) | MTEB Arena host a model arena for dynamic, real-world assessment of embedding models through user-based query and retrieval comparisons. |
| [MT-Bench-101](https://github.com/mtbench101/mt-bench-101?tab=readme-ov-file#leaderboard) | MT-Bench-101 is a fine-grained benchmark for evaluating LLMs in multi-turn dialogues. |
| [MY Malay LLM Leaderboard](https://huggingface.co/spaces/mesolitica/malay-llm-leaderboard) | MY Malay LLM Leaderboard aims to track, rank, and evaluate open LLMs on Malay tasks. |
| [NoCha](https://novelchallenge.github.io) | NoCha is a benchmark to evaluate how well long-context language models can verify claims written about fictional books. |
| [NPHardEval](https://huggingface.co/spaces/NPHardEval/NPHardEval-leaderboard) | NPHardEval is a benchmark to evaluate the reasoning abilities of LLMs through the lens of computational complexity classes. |
| [Occiglot Euro LLM Leaderboard](https://huggingface.co/spaces/occiglot/euro-llm-leaderboard) | Occiglot Euro LLM Leaderboard compares LLMs in four main languages from the Okapi benchmark and Belebele (French, Italian, German, Spanish and Dutch). |
| [OlympiadBench](https://github.com/OpenBMB/OlympiadBench?tab=readme-ov-file#leaderboard) | OlympiadBench is a bilingual multimodal scientific benchmark featuring 8,476 Olympiad-level mathematics and physics problems with expert-level step-by-step reasoning annotations. |
| [OlympicArena](https://gair-nlp.github.io/OlympicArena/#leaderboard) | OlympicArena is a benchmark to evaluate the advanced capabilities of LLMs across a broad spectrum of Olympic-level challenges. |
| [oobabooga](https://oobabooga.github.io/benchmark.html) | Oobabooga is a benchmark to perform repeatable performance tests of LLMs with the oobabooga web UI. |
| [OpenEval](http://openeval.org.cn/rank) | OpenEval is a platform assessto evaluate Chinese LLMs. |
| [OpenLLM Turkish leaderboard](https://huggingface.co/spaces/malhajar/OpenLLMTurkishLeaderboard) | OpenLLM Turkish leaderboard tracks progress and ranks the performance of LLMs in Turkish. |
| [Openness Leaderboard](https://huggingface.co/spaces/Shitqq/Openness-leaderboard) | Openness Leaderboard tracks and evaluates models' transparency in terms of open access to weights, data, and licenses, exposing models that fall short of openness standards. |
| [Openness Leaderboard](https://opening-up-chatgpt.github.io) | Openness Leaderboard is a tool that tracks the openness of instruction-tuned LLMs, evaluating their transparency, data, and model availability. |
| [OpenResearcher](https://github.com/GAIR-NLP/OpenResearcher?tab=readme-ov-file#-performance) | OpenResearcher contains the benchmarking results on various RAG-related systems as a leaderboard. |
| [Open Arabic LLM Leaderboard](https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard) | Open Arabic LLM Leaderboard tracks progress and ranks the performance of LLMs in Arabic. |
| [Open Chinese LLM Leaderboard](https://huggingface.co/spaces/BAAI/open_cn_llm_leaderboard) | Open Chinese LLM Leaderboard aims to track, rank, and evaluate open Chinese LLMs. |
| [Open CoT Leaderboard](https://huggingface.co/spaces/logikon/open_cot_leaderboard) | Open CoT Leaderboard tracks LLMs' abilities to generate effective chain-of-thought reasoning traces. |
| [Open Dutch LLM Evaluation Leaderboard](https://huggingface.co/spaces/BramVanroy/open_dutch_llm_leaderboard) | Open Dutch LLM Evaluation Leaderboard tracks progress and ranks the performance of LLMs in Dutch. |
| [Open Financial LLM Leaderboard](https://huggingface.co/spaces/TheFinAI/Open-Financial-LLM-Leaderboard) | Open Financial LLM Leaderboard aims to evaluate and compare the performance of financial LLMs. |
| [Open ITA LLM Leaderboard](https://huggingface.co/spaces/FinancialSupport/open_ita_llm_leaderboard) | Open ITA LLM Leaderboard tracks progress and ranks the performance of LLMs in Italian. |
| [Open Ko-LLM Leaderboard](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard) | Open Ko-LLM Leaderboard tracks progress and ranks the performance of LLMs in Korean. |
| [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) | Open LLM Leaderboard tracks progress and ranks the performance of LLMs in English. |
| [Open MLLM Leaderboard](https://huggingface.co/spaces/Wwwduojin/MLLM_leaderboard) | Open MLLM Leaderboard aims to track, rank and evaluate LLMs and chatbots. |
| [Open MOE LLM Leaderboard](https://huggingface.co/spaces/sparse-generative-ai/open-moe-llm-leaderboard) | OPEN MOE LLM Leaderboard assesses the performance and efficiency of various Mixture of Experts (MoE) LLMs. |
| [Open Multilingual LLM Evaluation Leaderboard](https://huggingface.co/spaces/uonlp/open_multilingual_llm_leaderboard) | Open Multilingual LLM Evaluation Leaderboard tracks progress and ranks the performance of LLMs in multiple languages. |
| [Open PL LLM Leaderboard](https://huggingface.co/spaces/speakleash/open_pl_llm_leaderboard) | Open PL LLM Leaderboard is a platform for assessing the performance of various LLMs in Polish. |
| [Open Portuguese LLM Leaderboard](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard) | Open PT LLM Leaderboard aims to evaluate and compare LLMs in the Portuguese-language tasks. |
| [Open Taiwan LLM leaderboard](https://huggingface.co/spaces/yentinglin/open-tw-llm-leaderboard) | Open Taiwan LLM leaderboard showcases the performance of LLMs on various Taiwanese Mandarin language understanding tasks. |
| [Open-LLM-Leaderboard](https://huggingface.co/spaces/Open-Style/OSQ-Leaderboard) | Open-LLM-Leaderboard evaluates LLMs in language understanding and reasoning by transitioning from multiple-choice questions (MCQs) to open-style questions. |
| [OPUS-MT Dashboard](https://opus.nlpl.eu/dashboard) | OPUS-MT Dashboard is a platform to track and compare machine translation models across multiple language pairs and metrics. |
| [OR-Bench](https://huggingface.co/spaces/bench-llms/or-bench-leaderboard) | OR-Bench is a benchmark to evaluate the over-refusal of enhanced safety in LLMs. |
| [ParsBench](https://huggingface.co/spaces/ParsBench/leaderboard) | ParsBench provides toolkits for benchmarking LLMs based on the Persian language. |
| [Persian LLM Leaderboard](https://huggingface.co/spaces/MatinaAI/persian_llm_leaderboard) | Persian LLM Leaderboard provides a reliable evaluation of LLMs in Persian Language. |
| [Pinocchio ITA leaderboard](https://huggingface.co/spaces/mii-llm/pinocchio-ita-leaderboard) | Pinocchio ITA leaderboard tracks and evaluates LLMs in Italian Language. |
| [PL-MTEB](https://huggingface.co/spaces/PL-MTEB/leaderboard) | PL-MTEB (Polish Massive Text Embedding Benchmark) is a benchmark for evaluating text embeddings in Polish across 28 NLP tasks. |
| [PromptBench](https://llm-eval.github.io/pages/leaderboard) | PromptBench is a benchmark to evaluate the robustness of LLMs on adversarial prompts. |
| [QAConv](https://github.com/salesforce/QAConv?tab=readme-ov-file#leaderboard) | QAConv is a benchmark for question answering using complex, domain-specific, and asynchronous conversations as the knowledge source. |
| [QuALITY](https://nyu-mll.github.io/quality) | QuALITY is a benchmark for evaluating multiple-choice question-answering with a long context. |
| [RABBITS](https://huggingface.co/spaces/AIM-Harvard/rabbits-leaderboard) | RABBITS is a benchmark to evaluate the robustness of LLMs by evaluating their handling of synonyms, specifically brand and generic drug names. |
| [Rakuda](https://yuzuai.jp/benchmark) | Rakuda is a benchmark to evaluate LLMs based on how well they answer a set of open-ended questions about Japanese topics. |
| [RedTeam Arena](https://redarena.ai/leaderboard) | RedTeam Arena is a red-teaming platform for LLMs. |
| [Red Teaming Resistance Benchmark](https://huggingface.co/spaces/HaizeLabs/red-teaming-resistance-benchmark) | Red Teaming Resistance Benchmark is a benchmark to evaluate the robustness of LLMs against red teaming prompts. |
| [ReST-MCTS*](https://github.com/THUDM/ReST-MCTS?tab=readme-ov-file#leaderboard) | ReST-MCTS* is a reinforced self-training method that uses tree search and process reward inference to collect high-quality reasoning traces for training policy and reward models without manual step annotations. |
| [Reviewer Arena](https://huggingface.co/spaces/openreviewer/reviewer-arena) | Reviewer Arena hosts the reviewer arena, where various LLMs compete based on their performance in critiquing academic papers. |
| [RoleEval](https://github.com/magnetic2014/roleeval?tab=readme-ov-file#leaderboard-5-shot) | RoleEval is a bilingual benchmark to evaluate the memorization, utilization, and reasoning capabilities of role knowledge of LLMs. |
| [RPBench Leaderboard](https://boson.ai/rpbench) | RPBench-Auto is an automated pipeline for evaluating LLMs using 80 personae for character-based and 80 scenes for scene-based role-playing. |
| [Russian Chatbot Arena](https://huggingface.co/spaces/LLMArena/llmarena-leaderboard) | Chatbot Arena hosts a chatbot arena where various LLMs compete in Russian based on user satisfaction. |
| [Russian SuperGLUE](https://russiansuperglue.com/leaderboard/2) | Russian SuperGLUE is a benchmark for Russian language models, focusing on logic, commonsense, and reasoning tasks. |
| [R-Judge](https://rjudgebench.github.io/leaderboard.html) | R-Judge is a benchmark to evaluate the proficiency of LLMs in judging and identifying safety risks given agent interaction records. |
| [Safety Prompts](http://coai.cs.tsinghua.edu.cn/leaderboard) | Safety Prompts is a benchmark to evaluate the safety of Chinese LLMs. |
| [SafetyBench](https://llmbench.ai/safety/data) | SafetyBench is a benchmark to evaluate the safety of LLMs. |
| [SALAD-Bench](https://huggingface.co/spaces/OpenSafetyLab/Salad-Bench-Leaderboard) | SALAD-Bench is a benchmark for evaluating the safety and security of LLMs. |
| [ScandEval](https://scandeval.com) | ScandEval is a benchmark to evaluate LLMs on tasks in Scandinavian languages as well as German, Dutch, and English. |
| [Science Leaderboard](https://huggingface.co/spaces/wenhu/Science-Leaderboard) | Science Leaderboard is a platform to evaluate LLMs' capabilities to solve science problems. |
| [SciGLM](https://github.com/THUDM/SciGLM?tab=readme-ov-file#leaderboard) | SciGLM is a suite of scientific language models that use a self-reflective instruction annotation framework to enhance scientific reasoning by generating and revising step-by-step solutions to unlabelled questions. |
| [SciKnowEval](http://scimind.ai/sciknoweval) | SciKnowEval is a benchmark to evaluate LLMs based on their proficiency in studying extensively, enquiring earnestly, thinking profoundly, discerning clearly, and practicing assiduously. |
| [SCROLLS](https://www.scrolls-benchmark.com/leaderboard) | SCROLLS is a benchmark to evaluate the reasoning capabilities of LLMs over long texts. |
| [SeaExam](https://huggingface.co/spaces/SeaLLMs/SeaExam_leaderboard) | SeaExam is a benchmark to evaluate LLMs for Southeast Asian (SEA) languages. |
| [SEAL LLM Leaderboards](https://scale.com/leaderboard) | SEAL LLM Leaderboards is an expert-driven private evaluation platform for LLMs. |
| [SeaEval](https://huggingface.co/spaces/SeaEval/SeaEval_Leaderboard) | SeaEval is a benchmark to evaluate the performance of multilingual LLMs in understanding and reasoning with natural language, as well as comprehending cultural practices, nuances, and values. |
| [SEA HELM](https://leaderboard.sea-lion.ai) | SEA HELM is a benchmark to evaluate LLMs' performance across English and Southeast Asian tasks, focusing on chat, instruction-following, and linguistic capabilities. |
| [SecEval](https://github.com/XuanwuAI/SecEval?tab=readme-ov-file#leaderboard) | SecEval is a benchmark to evaluate cybersecurity knowledge of foundation models. |
| [Self-Improving Leaderboard](https://huggingface.co/spaces/junkim100/self-improving-leaderboard) | Self-Improving Leaderboard (SIL) is a dynamic platform that continuously updates test datasets and rankings to provide real-time performance insights for open-source LLMs and chatbots. |
| [SimpleBench](https://simple-bench.com) | SimpleBench is a multiple-choice text benchmark where high school-level humans outperform all tested frontier LLMs, featuring 200+ questions on spatio-temporal reasoning, social intelligence, and linguistic adversarial robustness to test basic reasoning beyond memorized knowledge. |
| [Spec-Bench](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md) | Spec-Bench is a benchmark to evaluate speculative decoding methods across diverse scenarios. |
| [StructEval](https://huggingface.co/spaces/Bowieee/StructEval_leaderboard) | StructEval is a benchmark to evaluate LLMs by conducting structured assessments across multiple cognitive levels and critical concepts. |
| [Subquadratic LLM Leaderboard](https://huggingface.co/spaces/devingulliver/subquadratic-llm-leaderboard) | Subquadratic LLM Leaderboard evaluates LLMs with subquadratic/attention-free architectures (i.e. RWKV & Mamba). |
| [SuperBench](https://fm.ai.tsinghua.edu.cn/superbench) | SuperBench is a comprehensive system of tasks and dimensions to evaluate the overall capabilities of LLMs. |
| [SuperGLUE](https://super.gluebenchmark.com/leaderboard) | SuperGLUE is a benchmark to evaluate the performance of LLMs on a set of challenging language understanding tasks. |
| [SuperLim](https://lab.kb.se/leaderboard/results) | SuperLim is a benchmark to evaluate the language understanding capabilities of LLMs in Swedish. |
| [Swahili LLM-Leaderboard](https://github.com/msamwelmollel/Swahili_LLM_Leaderboard?tab=readme-ov-file#leaderboard) | Swahili LLM-Leaderboard is a joint community effort to create one central leaderboard for LLMs. |
| [S-Eval](https://huggingface.co/spaces/IS2Lab/S-Eval) | S-Eval is a comprehensive, multi-dimensional safety benchmark with 220,000 prompts designed to evaluate LLM safety across various risk dimensions. |
| [TableQAEval](https://github.com/lfy79001/TableQAKit?tab=readme-ov-file#leaderboard) | TableQAEval is a benchmark to evaluate LLM performance in modeling long tables and comprehension capabilities, such as numerical and multi-hop reasoning. |
| [TAT-DQA](https://nextplusplus.github.io/TAT-DQA) | TAT-DQA is a benchmark to evaluate LLMs on the discrete reasoning over documents that combine both structured and unstructured information. |
| [TAT-QA](https://nextplusplus.github.io/TAT-QA) | TAT-QA is a benchmark to evaluate LLMs on the discrete reasoning over documents that combines both tabular and textual content. |
| [Thai LLM Leaderboard](https://huggingface.co/spaces/ThaiLLM-Leaderboard/leaderboard) | Thai LLM Leaderboard aims to track and evaluate LLMs in the Thai-language tasks. |
| [The Pile](https://pile.eleuther.ai) | The Pile is a benchmark to evaluate the world knowledge and reasoning ability of LLMs. |
| [TOFU](https://huggingface.co/spaces/locuslab/tofu_leaderboard) | TOFU is a benchmark to evaluate the unlearning performance of LLMs in realistic scenarios. |
| [Toloka LLM Leaderboard](https://huggingface.co/spaces/toloka/open-llm-leaderboard) | Toloka LLM Leaderboard is a benchmark to evaluate LLMs based on authentic user prompts and expert human evaluation. |
| [Toolbench](https://huggingface.co/spaces/qiantong-xu/toolbench-leaderboard) | ToolBench is a platform for training, serving, and evaluating LLMs specifically for tool learning. |
| [Toxicity Leaderboard](https://huggingface.co/spaces/Bias-Leaderboard/leaderboard) | Toxicity Leaderboard evaluates the toxicity of LLMs. |
| [Trustbit LLM Leaderboards](https://www.trustbit.tech/en/llm-benchmarks) | Trustbit LLM Leaderboards is a platform that provides benchmarks for building and shipping products with LLMs. |
| [TrustLLM](https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html) | TrustLLM is a benchmark to evaluate the trustworthiness of LLMs. |
| [TuringAdvice](https://rowanzellers.com/advice) | TuringAdvice is a benchmark for evaluating language models' ability to generate helpful advice for real-life, open-ended situations. |
| [TutorEval](https://github.com/princeton-nlp/LM-Science-Tutor?tab=readme-ov-file#-leaderboard) | TutorEval is a question-answering benchmark which evaluates how well an LLM tutor can help a user understand a chapter from a science textbook. |
| [T-Eval](https://open-compass.github.io/T-Eval/leaderboard.html) | T-Eval is a benchmark for evaluating the tool utilization capability of LLMs. |
| [UGI Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard) | UGI Leaderboard measures and compares the uncensored and controversial information known by LLMs. |
| [UltraEval](https://ultraeval.openbmb.cn/rank) | UltraEval is an open-source framework for transparent and reproducible benchmarking of LLMs across various performance dimensions. |
| [VCR](https://visualcommonsense.com/leaderboard) | Visual Commonsense Reasoning (VCR) is a benchmark for cognition-level visual understanding, requiring models to answer visual questions and provide rationales for their answers. |
| [ViDoRe](https://huggingface.co/spaces/vidore/vidore-leaderboard) | ViDoRe is a benchmark to evaluate retrieval models on their capacity to match queries to relevant documents at the page level. |
| [VLLMs Leaderboard](https://huggingface.co/spaces/vlsp-2023-vllm/VLLMs-Leaderboard) | VLLMs Leaderboard aims to track, rank and evaluate open LLMs and chatbots. |
| [VMLU](https://vmlu.ai/leaderboard) | VMLU is a benchmark to evaluate overall capabilities of foundation models in Vietnamese. |
| [WildBench](https://huggingface.co/spaces/allenai/WildBench) | WildBench is a benchmark for evaluating language models on challenging tasks that closely resemble real-world applications. |
| [Xiezhi](https://github.com/MikeGu721/XiezhiBenchmark?tab=readme-ov-file#leaderboard) | Xiezhi is a benchmark for holistic domain knowledge evaluation of LLMs. |
| [Yanolja Arena](https://huggingface.co/spaces/yanolja/arena) | Yanolja Arena host a model arena to evaluate the capabilities of LLMs in summarizing and translating text. |
| [Yet Another LLM Leaderboard](https://huggingface.co/spaces/mlabonne/Yet_Another_LLM_Leaderboard) | Yet Another LLM Leaderboard is a platform for tracking, ranking, and evaluating open LLMs and chatbots. |
| [ZebraLogic](https://huggingface.co/spaces/allenai/ZebraLogic) | ZebraLogic is a benchmark evaluating LLMs' logical reasoning using Logic Grid Puzzles, a type of Constraint Satisfaction Problem (CSP). |
| [ZeroSumEval](https://huggingface.co/spaces/HishamYahya/ZeroSumEval_Leaderboard) | ZeroSumEval is a competitive evaluation framework for LLMs using multiplayer simulations with clear win conditions. |

### Code

| Name | Description |
| ---- | ----------- |
| [Aider LLM Leaderboards](https://aider.chat/docs/leaderboards) | Aider LLM Leaderboards evaluate LLM's ability to follow system prompts to edit code. |
| [AndroidWorld](https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo) | AndroidWorld is a comprehensive testing framework to evaluate the abilities of AI models, specifically autonomous agents, in controlling and interacting with a mobile device. |
| [AppWorld](https://github.com/stonybrooknlp/appworld/tree/main?tab=readme-ov-file#trophy-leaderboard) | AppWorld is a high-fidelity execution environment of 9 day-to-day apps, operable via 457 APIs, populated with digital activities of ~100 people living in a simulated world. |
| [Berkeley Function-Calling Leaderboard](https://huggingface.co/spaces/gorilla-llm/berkeley-function-calling-leaderboard) | Berkeley Function-Calling Leaderboard evaluates the ability of LLMs to call functions (also known as tools) accurately. |
| [BigCodeBench](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard) | BigCodeBench is a benchmark for code generation with practical and challenging programming tasks. |
| [Big Code Models Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard) | Big Code Models Leaderboard is a platform to track and evaluate the performance of LLMs on code-related tasks. |
| [BIRD](https://bird-bench.github.io) | BIRD is a benchmark to evaluate the performance of text-to-SQL parsing systems. |
| [BookSQL](https://huggingface.co/spaces/Exploration-Lab/BookSQL-Leaderboard) | BookSQL is a benchmark to evaluate Text-to-SQL systems in the finance and accounting domain across various industries with a dataset of 1 million transactions from 27 businesses. |
| [CanAiCode Leaderboard](https://huggingface.co/spaces/mike-ravkine/can-ai-code-results) | CanAiCode Leaderboard is a platform to evaluate the code generation capabilities of LLMs. |
| [ClassEval](https://fudanselab-classeval.github.io/leaderboard.html) | ClassEval is a benchmark to evaluate LLMs on class-level code generation. |
| [CodeApex](https://github.com/APEXLAB/CodeApex?tab=readme-ov-file#leaderboard) | CodeApex is a benchmark to evaluate LLMs' programming comprehension through multiple-choice questions and code generation with C++ algorithm problems. |
| [CodeScope](https://haitianliu22.github.io/code-scope-benchmark/leaderboard.html) | CodeScope is a benchmark to evaluate LLM coding capabilities across 43 languages and 8 tasks, considering difficulty, efficiency, and length. |
| [CodeTransOcean](https://yuchen814.github.io/CodeTransOcean/#leadboard) | CodeTransOcean is a benchmark to evaluate code translation across a wide variety of programming languages, including popular, niche, and LLM-translated code. |
| [Code Lingua](https://codetlingua.github.io/leaderboard.html) | Code Lingua is a benchmark to compare the ability of code models to understand what the code implements in source languages and translate the same semantics in target languages. |
| [Coding LLMs Leaderboard](https://leaderboard.tabbyml.com) | Coding LLMs Leaderboard is a platform to evaluate and rank LLMs across various programming tasks. |
| [Commit-0](https://commit-0.github.io/analysis) | Commit-0 is a from-scratch AI coding challenge to rebuild 54 core Python libraries, ensuring they pass unit tests with significant test coverage, lint/type checking, and cloud-based distributed development. |
| [CRUXEval](https://crux-eval.github.io/leaderboard.html) | CRUXEval is a benchmark to evaluate code reasoning, understanding, and execution capabilities of LLMs. |
| [CSpider](https://taolusi.github.io/CSpider-explorer) | CSpider is a benchmark to evaluate systems' ability to generate SQL queries from Chinese natural language across diverse, complex, and cross-domain databases. |
| [CyberSecEval](https://huggingface.co/spaces/facebook/CyberSecEval) | CyberSecEval is a benchmark to evaluate the cybersecurity of LLMs as coding assistants. |
| [DevEval](https://github.com/seketeam/DevEval?tab=readme-ov-file#leaderboard) | DevEval is a code generation benchmark collected through a rigorous pipeline. DevEval contains 1,825 testing samples, collected from 115 real-world code repositories and covering 10 programming topics. | 
| [DevOps AI Assistant Open Leaderboard](https://github.com/opstower-ai/devops-ai-open-leaderboard?tab=readme-ov-file#-current-leaderboard) | DevOps AI Assistant Open Leaderboard tracks, ranks, and evaluates DevOps AI Assistants across knowledge domains. |
| [DevOps-Eval](https://github.com/codefuse-ai/codefuse-devops-eval?tab=readme-ov-file#-leaderboard) | DevOps-Eval is a benchmark to evaluate code models in the DevOps/AIOps field. |
| [DomainEval](https://domaineval.github.io/leaderboard.html) | DomainEval is an auto-constructed benchmark for multi-domain code generation. |
| [Dr.Spider](https://github.com/awslabs/diagnostic-robustness-text-to-sql?tab=readme-ov-file#leaderboard) | Dr.Spider is a benchmark to evaluate the robustness of text-to-SQL models using different perturbation test sets. |
| [EffiBench](https://huggingface.co/spaces/EffiBench/effibench-leaderboard) | EffiBench is a benchmark to evaluate the efficiency of LLMs in code generation. |
| [EvalPlus](https://evalplus.github.io/leaderboard.html) | EvalPlus is a benchmark to evaluate the code generation performance of LLMs. |
| [EvoCodeBench](https://github.com/seketeam/EvoCodeBench?tab=readme-ov-file#leaderboard) | EvoCodeBench is an evolutionary code generation benchmark aligned with real-world code repositories. |
| [EvoEval](https://evo-eval.github.io/leaderboard.html) | EvoEval is a benchmark to evaluate the coding abilities of LLMs, created by evolving existing benchmarks into different targeted domains. |
| [InfiBench](https://infi-coder.github.io/infibench) | InfiBench is a benchmark to evaluate code models on answering freeform real-world code-related questions. |
| [InterCode](https://intercode-benchmark.github.io) | InterCode is a benchmark to standardize and evaluate interactive coding with execution feedback. |
| [Julia LLM Leaderboard](https://github.com/svilupp/Julia-LLM-Leaderboard?tab=readme-ov-file#results-preview) | Julia LLM Leaderboard is a platform to compare code models' abilities in generating syntactically correct Julia code, featuring structured tests and automated evaluations for easy and collaborative benchmarking. |
| [LiveCodeBench](https://livecodebench.github.io/leaderboard.html) | LiveCodeBench is a benchmark to evaluate code models across code-related scenarios over time. |
| [LiveCodeBench Pro](https://livecodebenchpro.com) | LiveCodeBench Pro evaluates LLMs on their ability to generate solutions for programming problems. The benchmark includes problems of varying difficulty levels from different competitive programming platforms. |
| [Long Code Arena](https://huggingface.co/spaces/JetBrains-Research/long-code-arena) | Long Code Arena is a suite of benchmarks for code-related tasks with large contexts, up to a whole code repository. |
| [McEval](https://mceval.github.io/leaderboard.html) | McEval is a massively multilingual code evaluation benchmark covering 40 languages (16K samples in 44 total), encompassing multilingual code generation, multilingual code explanation, and multilingual code completion tasks. |
| [Memorization or Generation of Big Code Models Leaderboard](https://huggingface.co/spaces/wzxii/Memorization-or-Generation-of-Big-Code-Models-Leaderboard) | Memorization or Generation of Big Code Models Leaderboard tracks and compares code generation models' performance. |
| [MLE-bench](https://github.com/openai/mle-bench?tab=readme-ov-file#leaderboard) | MLE-bench is a benchmark for measuring how well AI agents perform at machine learning engineering |
| [Multi-SWE-bench](https://multi-swe-bench.github.io) | Multi-SWE-bench is a multi-lingual GitHub issue resolving benchmark for code agents. |
| [NaturalCodeBench](https://github.com/THUDM/NaturalCodeBench?tab=readme-ov-file#leaderboard) | NaturalCodeBench is a benchmark to mirror the complexity and variety of scenarios in real coding tasks. |
| [Nexus Function Calling Leaderboard](https://huggingface.co/spaces/Nexusflow/Nexus_Function_Calling_Leaderboard) | Nexus Function Calling Leaderboard is a platform to evaluate code models on performing function calling and API usage.
| [NL2SQL360](https://nl2sql360.github.io/#leaderboard) | NL2SQL360 is a comprehensive evaluation framework for comparing and optimizing NL2SQL methods across various application scenarios. |
| [OSWorld](https://os-world.github.io) | OSWorld is a benchmark to evaluate multimodal AI agents on their ability to perform 369 realistic, open-ended tasks within a virtual computer environment across various applications and operating systems. |
| [PECC](https://huggingface.co/spaces/PatrickHaller/pecc-leaderboard) | PECC is a benchmark that evaluates code generation by requiring models to comprehend and extract problem requirements from narrative-based descriptions to produce syntactically accurate solutions. |
| [ProLLM Benchmarks](https://prollm.toqan.ai/leaderboard) | ProLLM Benchmarks is a practical and reliable LLM benchmark designed for real-world business use cases across multiple industries and programming languages. |
| [PyBench](https://github.com/Mercury7353/PyBench?tab=readme-ov-file#-leaderboard) | PyBench is a benchmark evaluating LLM on real-world coding tasks including chart analysis, text analysis, image/ audio editing, complex math and software/website development. |
| [RACE](https://huggingface.co/spaces/jszheng/RACE_leaderboard) | RACE is a benchmark to evaluate the ability of LLMs to generate code that is correct and meets the requirements of real-world development scenarios. |
| [RepairBench](https://repairbench.github.io) | RepairBench is a benchmark to evaluate the program repair capabilities of AI models by testing their ability to fix real-world software bugs. |
| [ResearchCodeBench](https://researchcodebench.github.io/leaderboard/index.html) | ResearchCodeBench is a benchmark for evaluating LLMs on their ability to translate novel machine learning research papers into executable code. |
| [RepoQA](https://evalplus.github.io/repoqa.html) | RepoQA is a benchmark to evaluate the long-context code understanding ability of LLMs.
| [ScreenSpot](https://gui-agent.github.io/grounding-leaderboard/screenspot.html) | ScreenSpot is a benchmark to evaluate the ability of models to perform GUI grounding across various platforms, including mobile (iOS, Android), desktop (macOS, Windows), and web environments, based on over 1,200 instructions. |
| [ScreenSpot-Pro](https://gui-agent.github.io/grounding-leaderboard) | ScreenSpot-Pro is a benchmark to evaluate the ability of multi-modal large language models (MLLMs) to accurately locate specific GUI elements in complex, high-resolution desktop applications. |
| [SciCode](https://github.com/scicode-bench/SciCode?tab=readme-ov-file#-leaderboard) | SciCode is a benchmark designed to evaluate language models in generating code to solve realistic scientific research problems. |
| [SE Arena](https://huggingface.co/spaces/SE-Arena/Software-Engineering-Arena) | SE Arena provides a platform for software developers to compare the performance of different FMs on software engineering tasks. |
| [SolidityBench](https://huggingface.co/spaces/braindao/solbench-leaderboard) | SolidityBench is a benchmark to evaluate and rank the ability of LLMs in generating and auditing smart contracts. |
| [Spider](https://yale-lily.github.io/spider) | Spider is a benchmark to evaluate the performance of natural language interfaces for cross-domain databases. |
| [StableToolBench](https://huggingface.co/spaces/stabletoolbench/Stable_Tool_Bench_Leaderboard) | StableToolBench is a benchmark to evaluate tool learning that aims to provide a well-balanced combination of stability and reality. |
| [SWE-bench](https://www.swebench.com) | SWE-bench is a benchmark for evaluating LLMs on real-world software issues collected from GitHub. |
| [SWE-bench-Live](https://swe-bench-live.github.io) | SWE-bench-Live is a live benchmark for issue resolving, designed to evaluate an AI system's ability to complete real-world software engineering tasks. |
| [Terminal-Bench](https://www.tbench.ai/leaderboard) | Terminal-Bench is a benchmark to measure the capabilities of AI agents in a terminal environment. |
| [UI-I2E-Bench](https://microsoft.github.io/FIVE-UI-Evol) | UI-I2E-Bench is a benchmark for GUI visual grounding. It incorporates implicit instructions and long-tail UI element types, with element-to-screen size ratios that better reflect real-world scenarios. |
| [VisualWebArena](https://jykoh.com/vwa) | VisualWebArena is a benchmark to evaluate the performance of multimodal web agents on realistic visually grounded tasks. |
| [WebAgent Leaderboard](https://huggingface.co/spaces/meghsn/WebAgent-Leaderboard) | WebAgent Leaderboard tracks and evaluates LLMs, VLMs, and agents on web navigation tasks. |
| [WebArena](https://docs.google.com/spreadsheets/d/1M801lEpBbKSNwP-vDBkC_pF7LdyGU1f_ufZb_NWNBZQ) | WebArena is a standalone, self-hostable web environment to evaluate autonomous agents. |
| [WebApp1K](https://huggingface.co/spaces/onekq-ai/WebApp1K-models-leaderboard) | WebApp1K is a benchmark to evaluate LLMs on their abilities to develop real-world web applications. |
| [WebDev Arena](https://web.lmarena.ai/leaderboard) | WebDev Arena hosts a chatbot arena where various LLMs compete based on website development. |
| [WILDS](https://wilds.stanford.edu/leaderboard) | WILDS is a benchmark of in-the-wild distribution shifts spanning diverse data modalities and applications, from tumor identification to wildlife monitoring to poverty mapping. |

## Image

| Name | Description |
| ---- | ----------- |
| [Abstract Image](https://multi-modal-self-instruct.github.io/#leaderboard) | Abstract Image is a benchmark to evaluate multimodal LLMs (MLLM) in understanding and visually reasoning about abstract images, such as maps, charts, and layouts. |
| [AesBench](https://aesbench.github.io) | AesBench is a benchmark to evaluate MLLMs on image aesthetics perception. |
| [BLINK](https://github.com/zeyofu/BLINK_Benchmark?tab=readme-ov-file#-mini-leaderboard) | BLINK is a benchmark to evaluate the core visual perception abilities of MLLMs. |
| [BlinkCode](https://huggingface.co/spaces/yajuniverse/BlinkCode_leaderboard) | BlinkCode is a benchmark to evaluate MLLMs across 15 vision-language models (VLMs) and 9 tasks, measuring accuracy and image reconstruction performance. |
| [ChartMimic](https://chartmimic.github.io) | ChartMimic is a benchmark to evaluate the visually-grounded code generation capabilities of large multimodal models using charts and textual instructions. |
| [CharXiv](https://charxiv.github.io/#leaderboard) | CharXiv is a benchmark to evaluate chart understanding capabilities of MLLMs. |
| [ConTextual](https://huggingface.co/spaces/ucla-contextual/contextual_leaderboard) | ConTextual is a benchmark to evaluate MLLMs across context-sensitive text-rich visual reasoning tasks. |
| [CORE-MM](https://core-mm.github.io) | CORE-MM is a benchmark to evaluate the open-ended visual question-answering (VQA) capabilities of MLLMs. |
| [DreamBench++](https://dreambenchplus.github.io/#leaderboard) | DreamBench++ is a human-aligned benchmark automated by multimodal models for personalized image generation. |
| [EgoPlan-Bench](https://huggingface.co/spaces/ChenYi99/EgoPlan-Bench_Leaderboard) | EgoPlan-Bench is a benchmark to evaluate planning abilities of MLLMs in real-world, egocentric scenarios. |
| [HallusionBench](https://github.com/tianyi-lab/HallusionBench?tab=readme-ov-file#leaderboard) | HallusionBench is a benchmark to evaluate the image-context reasoning capabilities of MLLMs. |
| [InfiMM-Eval](https://infimm.github.io/InfiMM-Eval) | InfiMM-Eval is a benchmark to evaluate the open-ended VQA capabilities of MLLMs. |
| [LRVSF Leaderboard](https://huggingface.co/spaces/Slep/LRVSF-Leaderboard) | LRVSF Leaderboard is a platform to evaluate LLMs regarding image similarity search in fashion. |
| [LVLM Leaderboard](https://github.com/OpenGVLab/Multi-Modality-Arena?tab=readme-ov-file#lvlm-leaderboard) | LVLM Leaderboard is a platform to evaluate the visual reasoning capabilities of MLLMs. |
| [M3CoT](https://lightchen233.github.io/m3cot.github.io/leaderboard.html) | M3CoT is a benchmark for multi-domain multi-step multi-modal chain-of-thought of MLLMs. |
| [Mementos](https://mementos-bench.github.io/#leaderboard) | Mementos is a benchmark to evaluate the reasoning capabilities of MLLMs over image sequences. |
| [MJ-Bench](https://huggingface.co/spaces/MJ-Bench/MJ-Bench-Leaderboard) | MJ-Bench is a benchmark to evaluate multimodal judges in providing feedback for image generation models across four key perspectives: alignment, safety, image quality, and bias. |
| [MLLM-as-a-Judge](https://mllm-judge.github.io/leaderboard.html) | MLLM-as-a-Judge is a benchmark with human annotations to evaluate MLLMs' judging capabilities in scoring, pair comparison, and batch ranking tasks across multimodal domains. |
| [MLLM-Bench](https://mllm-bench.llmzoo.com/static/leaderboard.html) | MLLM-Bench is a benchmark to evaluate the visual reasoning capabilities of MLVMs. |
| [MMBench Leaderboard](https://mmbench.opencompass.org.cn/leaderboard) | MMBench Leaderboard is a platform to evaluate the visual reasoning capabilities of MLLMs. |
| [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation?tab=readme-ov-file#leaderboards-of-available-models-unavailable-version) | MME is a benchmark to evaluate the visual reasoning capabilities of MLLMs. |
| [MME-RealWorld](https://mme-realworld.github.io/home_page.html#leaderboard) | MME-RealWorld is a large-scale, high-resolution benchmark featuring 29,429 human-annotated QA pairs across 43 tasks. |
| [MMIU](https://mmiu-bench.github.io/#leaderboard) | MMIU (Ultimodal Multi-image Understanding) is a benchmark to evaluate MLLMs across 7 multi-image relationships, 52 tasks, 77K images, and 11K curated multiple-choice questions. |
| [MMMU](https://mmmu-benchmark.github.io/#leaderboard) | MMMU is a benchmark to evaluate the performance of multimodal models on tasks that demand college-level subject knowledge and expert-level reasoning across various disciplines. |
| [MMR](https://github.com/BAAI-DCAI/Multimodal-Robustness-Benchmark?tab=readme-ov-file#-leaderboard) | MMR is a benchmark to evaluate the robustness of MLLMs in visual understanding by assessing their ability to handle leading questions, rather than just accuracy in answering. |
| [MMSearch](https://mmsearch.github.io/#leaderboard) | MMSearch is a benchmark to evaluate the multimodal search performance of LMMs. |
| [MMStar](https://mmstar-benchmark.github.io/#Leaderboard) | MMStar is a benchmark to evaluate the multi-modal capacities of MLLMs. |
| [MMT-Bench](https://mmt-bench.github.io/#leaderboard) | MMT-Bench is a benchmark to evaluate MLLMs across a wide array of multimodal tasks that require expert knowledge as well as deliberate visual recognition, localization, reasoning, and planning. |
| [MM-NIAH](https://mm-niah.github.io/#overall_test_leaderboard) | MM-NIAH (Needle In A Multimodal Haystack) is a benchmark to evaluate MLLMs' ability to comprehend long multimodal documents through retrieval, counting, and reasoning tasks involving both text and image data. |
| [MTVQA](https://bytedance.github.io/MTVQA/#leaderboard) | MTVQA is a multilingual visual text comprehension benchmark to evaluate MLLMs. |
| [Multimodal Hallucination Leaderboard](https://huggingface.co/spaces/scb10x/multimodal-hallucination-leaderboard) | Multimodal Hallucination Leaderboard compares MLLMs based on hallucination levels in various tasks. |
| [MULTI-Benchmark](https://github.com/OpenDFM/MULTI-Benchmark?tab=readme-ov-file#-leaderboard) | MULTI-Benchmark is a benchmark to evaluate MLLMs on understanding complex tables and images, and reasoning with long context. |
| [MultiTrust](https://multi-trust.github.io/#leaderboard) | MultiTrust is a benchmark to evaluate the trustworthiness of MLLMs across five primary aspects: truthfulness, safety, robustness, fairness, and privacy. |
| [NPHardEval4V](https://github.com/lizhouf/nphardeval4v?tab=readme-ov-file#leaderboard) | NPHardEval4V is a benchmark to evaluate the reasoning abilities of MLLMs through the lens of computational complexity classes. |
| [Provider Leaderboard](https://leaderboard.withmartian.com) | LLM API Providers Leaderboard is a platform to compare API provider performance for over LLM endpoints across performance key metrics. |
| [OCRBench](https://huggingface.co/spaces/echo840/ocrbench-leaderboard) | OCRBench is a benchmark to evaluate the OCR capabilities of multimodal models. |
| [PCA-Bench](https://github.com/pkunlp-icler/PCA-EVAL?tab=readme-ov-file#leaderboard) | PCA-Bench is a benchmark to evaluate the embodied decision-making capabilities of multimodal models. |
| [Q-Bench](https://huggingface.co/spaces/q-future/Q-Bench-Leaderboard) | Q-Bench is a benchmark to evaluate the visual reasoning capabilities of MLLMs. |
| [RewardBench](https://huggingface.co/spaces/allenai/reward-bench) | RewardBench is a benchmark to evaluate the capabilities and safety of reward models. |
| [ScienceQA](https://scienceqa.github.io/leaderboard.html) | ScienceQA is a benchmark used to evaluate the multi-hop reasoning ability and interpretability of AI systems in the context of answering science questions. |
| [SciGraphQA](https://github.com/findalexli/SciGraphQA?tab=readme-ov-file#leaderboard) | SciGraphQA is a benchmark to evaluate the MLLMs in scientific graph question-answering. |
| [SEED-Bench](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) | SEED-Bench is a benchmark to evaluate the text and image generation of multimodal models. |
| [URIAL](https://huggingface.co/spaces/allenai/URIAL-Bench) | URIAL is a benchmark to evaluate the capacity of language models for alignment without introducing the factors of fine-tuning (learning rate, data, etc.), which are hard to control for fair comparisons. |
| [UPD Leaderboard](https://huggingface.co/spaces/MM-UPD/MM-UPD_Leaderboard) | UPD Leaderboard is a platform to evaluate the trustworthiness of MLLMs in unsolvable problem detection. |
| [Vibe-Eval](https://github.com/reka-ai/reka-vibe-eval?tab=readme-ov-file#leaderboard-) | Vibe-Eval is a benchmark to evaluate MLLMs for challenging cases. |
| [VideoHallucer](https://videohallucer.github.io) | VideoHallucer is a benchmark to detect hallucinations in MLLMs. |
| [VisIT-Bench](https://visit-bench.github.io) | VisIT-Bench is a benchmark to evaluate the instruction-following capabilities of MLLMs for real-world use. |
| [Waymo Open Dataset Challenges](https://waymo.com/open/challenges) | Waymo Open Dataset Challenges hold diverse self-driving datasets to evaluate ML models. |
| [WHOOPS!](https://huggingface.co/spaces/nlphuji/WHOOPS-Leaderboard-Full) | WHOOPS! is a benchmark to evaluate the visual commonsense reasoning abilities of MLLMs. |
| [WildVision-Bench](https://github.com/WildVision-AI/WildVision-Bench?tab=readme-ov-file#leaderboard-vision_bench_0617) | WildVision-Bench is a benchmark to evaluate VLMs in the wild with human preferences. |
| [WildVision Arena](https://huggingface.co/spaces/WildVision/vision-arena) | WildVision Arena hosts the chatbot arena where various MLLMs compete based on their performance in visual understanding. |

### Video

| Name | Description |
| ---- | ----------- |
| [ChronoMagic-Bench](https://huggingface.co/spaces/BestWishYsh/ChronoMagic-Bench) | ChronoMagic-Bench is a benchmark to evaluate video models' ability to generate time-lapse videos with high metamorphic amplitude and temporal coherence across physics, biology, and chemistry domains using free-form text control. |
| [DREAM-1K](https://tarsier-vlm.github.io) | DREAM-1K is a benchmark to evaluate video description performance on 1,000 diverse video clips featuring rich events, actions, and motions from movies, animations, stock videos, YouTube, and TikTok-style short videos. |
| [LongVideoBench](https://huggingface.co/spaces/longvideobench/LongVideoBench) | LongVideoBench is a benchmark to evaluate the capabilities of video models in answering referred reasoning questions, which are dependent on long frame inputs and cannot be well-addressed by a single frame or a few sparse frames. |
| [LVBench](https://lvbench.github.io) | LVBench is a benchmark to evaluate multimodal models on long video understanding tasks requiring extended memory and comprehension capabilities. |
| [MLVU](https://github.com/JUNJIE99/MLVU?tab=readme-ov-file#trophy-mini-leaderboard-mlvu-dev-set) | MLVU is a benchmark to evaluate video models in multi-task long video understanding. |
| [MMToM-QA](https://chuanyangjin.com/mmtom-qa-leaderboard) | MMToM-QA is a multimodal benchmark to evaluate machine Theory of Mind (ToM), the ability to understand people's minds. |
| [MVBench](https://huggingface.co/spaces/OpenGVLab/MVBench_Leaderboard) | MVBench is a benchmark to evaluate the temporal understanding capabilities of video models in dynamic video tasks. |
| [OpenVLM Video Leaderboard](https://huggingface.co/spaces/opencompass/openvlm_video_leaderboard) | OpenVLM Video Leaderboard is a platform showcasing the evaluation results of 30 different VLMs on video understanding benchmarks using the VLMEvalKit framework. |
| [TempCompass](https://huggingface.co/spaces/lyx97/TempCompass) | TempCompass is a benchmark to evaluate Video LLMs' temporal perception using 410 videos and 7,540 task instructions across 11 temporal aspects and 4 task types. |
| [VBench](https://vchitect.github.io/VBench-project) | VBench is a benchmark to evaluate video generation capabilities of video models. |
| [VideoNIAH](https://videoniah.github.io/#leaderboard) | VideoNIAH is a benchmark to evaluate the fine-grained understanding and spatio-temporal modeling capabilities of video models. |
| [VideoPhy](https://github.com/Hritikbansal/videophy) | VideoPhy is a benchmark to evaluate generated videos for adherence to physical commonsense in real-world material interactions. |
| [VideoScore](https://huggingface.co/spaces/TIGER-Lab/VideoScore-Leaderboard) | VideoScore is a benchmark to evaluate text-to-video generative models on five key dimensions. |
| [VideoVista](https://videovista.github.io/#leaderboard) | VideoVista is a benchmark with 25,000 questions from 3,400 videos across 14 categories, covering 19 understanding and 8 reasoning tasks. |
| [Video-Bench](https://huggingface.co/spaces/LanguageBind/Video-Bench) | Video-Bench is a benchmark to evaluate the video-exclusive understanding, prior knowledge incorporation, and video-based decision-making abilities of video models. |
| [Video-MME](https://video-mme.github.io/home_page.html#leaderboard) | Video-MME is a benchmark to evaluate the video analysis capabilities of video models. |

### Math

| Name | Description |
| ---- | ----------- |
| [Abel](https://gair-nlp.github.io/abel) | Abel is a platform to evaluate the mathematical capabilities of LLMs. |
| [MathArena](https://matharena.ai) | MathArena is a platform for evaluation of LLMs on the latest math competitions and olympiads. |
| [MathBench](https://open-compass.github.io/MathBench) | MathBench is a multi-level difficulty mathematics evaluation benchmark for LLMs. |
| [MathEval](https://matheval.ai/leaderboard) | MathEval is a benchmark to evaluate the mathematical capabilities of LLMs. |
| [MathUserEval](https://github.com/THUDM/ChatGLM-Math?tab=readme-ov-file#%E6%8E%92%E8%A1%8C%E6%A6%9Cleaderboard) | MathUserEval is a benchmark featuring university exam questions and math-related queries derived from simulated conversations with experienced annotators. |
| [MathVerse](https://mathverse-cuhk.github.io/#leaderboard) | MathVerse is a benchmark to evaluate vision-language models in interpreting and reasoning with visual information in mathematical problems. |
| [MathVista](https://mathvista.github.io/#leaderboard) | MathVista is a benchmark to evaluate mathematical reasoning in visual contexts. |
| [MATH-V](https://mathvision-cuhk.github.io) | MATH-Vision (MATH-V) is a benchmark of 3,040 visually contextualized math problems from competitions, covering 16 disciplines and 5 difficulty levels to evaluate LMMs' mathematical reasoning. |
| [Open Multilingual Reasoning Leaderboard](https://huggingface.co/spaces/kevinpro/Open-Multilingual-Reasoning-Leaderboard) | Open Multilingual Reasoning Leaderboard tracks and ranks the reasoning performance of LLMs on multilingual mathematical reasoning benchmarks. |
| [PutnamBench](https://trishullab.github.io/PutnamBench/leaderboard.html) | PutnamBench is a benchmark to evaluate the formal mathematical reasoning capabilities of LLMs on the Putnam Competition. |
| [SciBench](https://scibench-ucla.github.io/#leaderboard) | SciBench is a benchmark to evaluate the reasoning capabilities of LLMs for solving complex scientific problems. |
| [TabMWP](https://promptpg.github.io/leaderboard.html) | TabMWP is a benchmark to evaluate LLMs in mathematical reasoning tasks that involve both textual and tabular data. |
| [We-Math](https://we-math.github.io/#leaderboard) | We-Math is a benchmark to evaluate the human-like mathematical reasoning capabilities of LLMs with problem-solving principles beyond the end-to-end performance. |

### Agent

| Name | Description |
| ---- | ----------- |
| [AgentBench](https://llmbench.ai/agent/data) | AgentBench is the benchmark to evaluate language model-as-Agent across a diverse spectrum of different environments. |
| [AgentBoard](https://hkust-nlp.github.io/agentboard/static/leaderboard.html) | AgentBoard is a benchmark for multi-turn LLM agents, complemented by an analytical evaluation board for detailed model assessment beyond final success rates. |
| [AgentStudio](https://skyworkai.github.io/agent-studio) | AgentStudio is an integrated solution featuring in-depth benchmark suites, realistic environments, and comprehensive toolkits. |
| [BrowserGym Leaderboard](https://huggingface.co/spaces/ServiceNow/browsergym-leaderboard) | BrowserGym Leaderboard is a platform to evaluate LLMs, VLMs, and agents on web navigation tasks. |
| [CharacterEval](https://github.com/morecry/CharacterEval?tab=readme-ov-file#leaderboard) | CharacterEval is a benchmark to evaluate Role-Playing Conversational Agents (RPCAs) using multi-turn dialogues and character profiles, with metrics spanning four dimensions. |
| [GTA](https://github.com/open-compass/GTA?tab=readme-ov-file#leaderboard) | GTA is a benchmark to evaluate the tool-use capability of LLM-based agents in real-world scenarios.
| [Leetcode-Hard Gym](https://github.com/GammaTauAI/leetcode-hard-gym?tab=readme-ov-file#leaderboard-for-leetcode-hard-python-pass1) | Leetcode-Hard Gym is an RL environment interface to LeetCode's submission server for evaluating codegen agents. |
| [LLM Colosseum Leaderboard](https://github.com/OpenGenerativeAI/llm-colosseum?tab=readme-ov-file#results) | LLM Colosseum Leaderboard is a platform to evaluate LLMs by fighting in Street Fighter 3. |
| [MAgIC](https://github.com/cathyxl/MAgIC?tab=readme-ov-file#leaderboard) | MAgIC is a benchmark to measure the abilities of cognition, adaptability, rationality and collaboration of LLMs within multi-agent sytems. |
| [Olas Predict Benchmark](https://huggingface.co/spaces/valory/olas-prediction-leaderboard) | Olas Predict Benchmark is a benchmark to evaluate agents on historical and future event forecasting. |
| [TravelPlanner](https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard) | TravelPlanner is a benchmark to evaluate LLM agents in tool use and complex planning within multiple constraints. |
| [VAB](https://github.com/THUDM/VisualAgentBench?tab=readme-ov-file#leaderboard) | VisualAgentBench (VAB) is a benchmark to evaluate and develop LMMs as visual foundation agents, which comprises 5 distinct environments across 3 types of representative visual agent tasks. |
| [τ-Bench](https://github.com/sierra-research/tau-bench?tab=readme-ov-file#leaderboard) | τ-bench is a benchmark that emulates dynamic conversations between a language model-simulated user and a language agent equipped with domain-specific API tools and policy guidelines. |

### Medical

| Name | Description |
| ---- | ----------- |
| [CARES](https://cares-ai.github.io/#leaderboard) | CARES is a benchmark to evaluate the trustworthiness of Med-LVLMs across trustfulness, fairness, safety, privacy, and robustness using 41K question-answer pairs from 16 medical image modalities and 27 anatomical regions. |
| [CMB](https://cmedbenchmark.llmzoo.com/static/leaderboard.html) | CMB is a multi-level medical benchmark in Chinese. |
| [JMED-LLM](https://github.com/sociocom/JMED-LLM?tab=readme-ov-file#leaderboard) | JMED-LLM (Japanese Medical Evaluation Dataset for Large Language Models) is a benchmark for evaluating LLMs in the medical field of Japanese. |
| [Mirage](https://teddy-xionggz.github.io/MIRAGE) | Mirage is a benchmark for medical information retrieval-augmented generation, featuring 7,663 questions from five medical QA datasets and tested with 41 configurations using the MedRag toolkit. |
| [MedArena](https://medarena.ai/leaderboard) | MedArena provides a platform for clinicians to compare the performance of different LLMs on clinical tasks. |
| [MedBench](https://medbench.opencompass.org.cn/leaderboard) | MedBench is a benchmark to evaluate the mastery of knowledge and reasoning abilities in medical LLMs. |
| [MedS-Bench](https://henrychur.github.io/MedS-Bench) | MedS-Bench is a medical benchmark that evaluates LLMs across 11 task categories using 39 diverse datasets. |
| [MMedBench](https://henrychur.github.io/MultilingualMedQA) | MMedBench is a medical benchmark to evaluate LLMs in multilingual comprehension. |
| [Open Medical-LLM Leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard) | Open Medical-LLM Leaderboard aims to track, rank, and evaluate open LLMs in the medical domain. |
| [Polish Medical Leaderboard](https://huggingface.co/spaces/speakleash/polish_medical_leaderboard) | Polish Medical Leaderboard evaluates language models on Polish board certification examinations. |
| [Powered-by-Intel LLM Leaderboard](https://huggingface.co/spaces/Intel/powered_by_intel_llm_leaderboard) | Powered-by-Intel LLM Leaderboard evaluates, scores, and ranks LLMs that have been pre-trained or fine-tuned on Intel Hardware. |
| [PubMedQA](https://pubmedqa.github.io) | PubMedQA is a benchmark to evaluate biomedical research question answering. |

### Audio

| Name | Description |
| ---- | ----------- |
| [AIR-Bench](https://github.com/OFA-Sys/AIR-Bench?tab=readme-ov-file#leaderboard) | AIR-Bench is a benchmark to evaluate the ability of audio models to understand various types of audio signals (including human speech, natural sounds and music), and furthermore, to interact with humans in textual format. |
| [AudioBench](https://huggingface.co/spaces/AudioLLMs/AudioBench-Leaderboard) | AudioBench is a benchmark for general instruction-following audio models. |
| [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) | Open ASR Leaderboard provides a platform for tracking, ranking, and evaluating Automatic Speech Recognition (ASR) models. |
| [Polish ASR Leaderboard](https://huggingface.co/spaces/amu-cai/pl-asr-leaderboard) | Polish ASR leaderboard aims to provide comprehensive overview of performance of ASR/STT systems for Polish. |
| [SALMon](https://pages.cs.huji.ac.il/adiyoss-lab/salmon) | SALMon is an evaluation suite that benchmarks speech language models on consistency, background noise, emotion, speaker identity, and room impulse response. |
| [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) | TTS-Arena hosts the Text To Speech (TTS) arena, where various TTS models compete based on their performance in generating speech. |
| [Whisper Leaderboard](https://huggingface.co/spaces/philipp-zettl/whisper-leaderboard) | Whisper Leaderboard is a platform tracking and comparing audio models' speech recognition performance on various datasets. |

### 3D

| Name | Description |
| ---- | ----------- |
| [3D Arena](https://huggingface.co/spaces/dylanebert/3d-arena) | 3D Arena hosts 3D generation arena, where various 3D generative models compete based on their performance in generating 3D models. |
| [3D-POPE](https://huggingface.co/spaces/sled-umich/3D-POPE-leaderboard) | 3D-POPE is a benchmark to evaluate object hallucination in 3D generative models. |
| [3DGen Arena](https://huggingface.co/spaces/ZhangYuhan/3DGen-Arena) | 3DGen Arena hosts the 3D generation arena, where various 3D generative models compete based on their performance in generating 3D models. |
| [BOP](https://bop.felk.cvut.cz/leaderboards) | BOP is a benchmark for 6D pose estimation of a rigid object from a single RGB-D input image. |
| [GPTEval3D](https://huggingface.co/spaces/GPTEval3D/Leaderboard_dev) | GPTEval3D is a benchmark to evaluate MLLMs' capabiltiies of 3D content understanding via multi-view images as input. |

### Game

| Name | Description |
| ---- | ----------- |
| [γ-Bench](https://github.com/CUHK-ARISE/GAMABench?tab=readme-ov-file#leaderboard) | γ-Bench is a framework for evaluating LLMs' gaming abilities in multi-agent environments using eight classical game theory scenarios and a dynamic scoring scheme. |
| [Elo Leaderboard](https://werewolf.foaster.ai) | Elo Leaderboard is a platform to evaluate LLMs' ability to deceive, deduce, and form alliances in the classic social deduction game of Werewolf, revealing which models excel at strategic reasoning and social manipulation. |
| [GlitchBench](https://huggingface.co/spaces/glitchbench/Leaderboard) | GlitchBench is a benchmark to evaluate the reasoning capabilities of MLLMs in the context of detecting video game glitches. |
| [LLM Roleplay Leaderboard](https://huggingface.co/spaces/hackathonM/Roleplay_leaderboard) | LLM Roleplay Leaderboard evaluates human and AI performance in a social werewolf game for NPC development. |
| [LMGame Bench](https://huggingface.co/spaces/lmgame/lmgame_bench) | LMGame Bench is a benchmark for evaluating large language models' performance on game-playing tasks and strategic reasoning abilities. |
| [GTBench](https://huggingface.co/spaces/GTBench/GTBench) | GTBench is a benchmark to evaluate and rank LLMs' reasoning abilities in competitive environments through game-theoretic tasks, e.g., board and card games. |

### Multimodal

| Name | Description |
| ---- | ----------- |
| [AV-Odyssey](https://av-odyssey.github.io/home_page.html) | AV-Odyssey Bench is a comprehensive audio-visual benchmark to assess whether those MLLMs can truly understand the audio-visual information. |
| [GenAI Arena](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena) | GenAI Arena hosts the visual generation arena, where various vision models compete based on their performance in image generation, image edition, and video generation. |
| [Labelbox Leaderboards](https://labelbox.com/leaderboards) | Labelbox Leaderboards evaluate performance of generative AI models using their data factory: platform, scientific process and expert humans.
| [MEGA-Bench](https://huggingface.co/spaces/TIGER-Lab/MEGA-Bench) | MEGA-Bench is a benchmark for multimodal evaluation with diverse tasks across 8 application types, 7 input formats, 6 output formats, and 10 multimodal skills, spanning single-image, multi-image, and video tasks. |

### Intelligence Quotient

| Name | Description |
| ---- | ----------- |
| [Tracking AI leaderboard](https://www.trackingai.org) | Tracking AI leaderboard is an online ranking platform of AI models based on their performance on various IQ-style evaluations. |

## Database Ranking

| Name | Description |
| ---- | ----------- |
| [VectorDBBench](https://zilliz.com/vector-database-benchmark-tool) | VectorDBBench is a benchmark to evaluate performance, cost-effectiveness, and scalability of various vector databases and cloud-based vector database services. |

## Dataset Ranking

| Name | Description |
| ---- | ----------- |
| [DataComp](https://www.datacomp.ai) | DataComp is a benchmark to evaluate the performance of various datasets with a fixed model architecture. |

## Metric Ranking

| Name | Description |
| ---- | ----------- |
| [AlignScore](https://github.com/yuh-zha/AlignScore?tab=readme-ov-file#leaderboards) | AlignScore evaluates the performance of different metrics in assessing factual consistency. |

## Paper Ranking

| Name | Description |
| ---- | ----------- |
| [Papers Leaderboard](https://huggingface.co/spaces/ameerazam08/Paper-LeaderBoard) | Papers Leaderboard is a platform to evaluate the popularity of machine learning papers. |

## Leaderboard Ranking

| Name | Description |
| ---- | ----------- |
| [Open Leaderboards Leaderboard](https://huggingface.co/spaces/mrfakename/open-leaderboards-leaderboard) | Open Leaderboards Leaderboard is a meta-leaderboard that leverages human preferences to compare machine learning leaderboards. |
