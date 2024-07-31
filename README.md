```markdown
# Pegasus WMS LLM Assistant

## Overview

This project aims to develop a Large Language Model (LLM) specifically trained to assist users with Pegasus Workflow Management System (WMS) documentation and workflow creation. The model is designed to provide accurate responses to queries about Pegasus WMS and help in writing Pegasus workflows.

## Project Goals

1. Fine-tune LLMs on Pegasus WMS documentation
2. Implement a Retrieval-Augmented Generation (RAG) system for accurate information retrieval
3. Develop an interactive interface for querying about Pegasus WMS and getting assistance in workflow creation

## Project Structure

```bash
.
├── FinetuneLLmPegasus
│   ├── MainFineTune.ipynb
│   └── finetunellamaDraft.ipynb
├── OnlineWebDoc
│   └── doc.py
├── RAGOnline
│   └── index.py
├── data.json
├── requirements.txt
└── workflow
    ├── Dockerfile
    ├── bin
    │   └── finetune.py
    ├── data
    │   └── data.json
    ├── generated_workflows
    │   ├── Llama-2-7b-chat-hf.yml
    │   ├── Meta-Llama-3-70B-Instruct.yml
    │   ├── Meta-Llama-3-8B-Instruct.yml
    │   ├── Mistral-7B-Instruct-v0.3.yml
    │   └── falcon-7b.yml
    ├── pegasus.properties
    ├── run_workdlows.sh
    └── workflow.py
```

## Key Components

1. **FinetuneLLmPegasus**: Notebooks for fine-tuning LLMs on Pegasus WMS documentation.
2. **RAGOnline**: Implementation of RAG for accurate information retrieval from Pegasus docs.
3. **workflow**: Contains scripts and configurations for generating Pegasus workflows to finetune LLM.
4. **OnlineWebDoc**: Web interface for interacting with the  -model with LLM connected to internet and specifically Pegasus-wms documentation.

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/swarmourr/Pegasus-wm-llm.git
   cd pegasus-wms-llm
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download and prepare Pegasus WMS documentation data (not included in repo).

## Usage

### Fine-tuning LLM

1. Place Pegasus WMS documentation data in `data/pegasus_docs.json`.
2. Run the fine-tuning notebook:
   ```
   jupyter notebook FinetuneLLmPegasus/MainFineTune.ipynb
   ```

### Running RAG System

1. Ensure the fine-tuned model is saved in the appropriate directory.
2. Start the RAG system:
   ```
   python RAGOnline/index.py
   ```

### Generating Workflows

1. Navigate to the `workflow` directory.
2. Use the workflow generation script:
   ```
   python workflow.py --help
   ```

### Web Interface

1. Start the web interface:
   ```
   python OnlineWebDoc/doc.py
   ```
2. Open a browser and go to `http://localhost:5000`.

## Model Information

We experiment with various LLMs including:
- Llama-2-7b-chat-hf
- Meta-Llama-3-70B-Instruct
- Meta-Llama-3-8B-Instruct
- Mistral-7B-Instruct-v0.3
- falcon-7b

These models are fine-tuned on Pegasus WMS documentation for optimal performance.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to the branch.
5. Create a new Pull Request.



