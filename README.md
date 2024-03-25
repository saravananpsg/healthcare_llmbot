# LLM Healthcare chatbot using Openai GPT3.5 retriever model w/ chaining:

## Project objective: 
This repository is to create a conversational chatbot that can address readers’ concerns in natural language using information from the trusted articles and in the  healthcare context.

## Proposed idea:
* Using Langchain and Openai model to retrieve top-k candidates from the online sources and pass it to Openai model to find the answer for the query

## Implementation

### 1 Pipeline: 

* Reading Documents: Data from Online PDF files and html files are extracted and loaded to Document loaders
* Chunking:  After loading, data must be split into smaller chunks that can be fit into the LLM model for further processing
  * RecursiveCharacterTextSplitter: Recommended splitter for generic text. It tries to split into blocks based on the following order *[“\n\n”, “\n”, “ “, “”]*
* Create a Vector Database: 
  * Unstructured data is commonly stored by projecting it into an embedding vector space, which provides a real number to each encoded vector. 
  * All the chunks are converted to embedding vector and stored in vector db. 
    * Used FAISS which is efficient for similarity search and clustering of dense vectors
* Search (once per query)
  * Given a user question, generate an embedding for the query from the OpenAI API
  * Using the embeddings, rank the text sections by relevance to the query
* Ask (once per query)
  * Insert the question and the most relevant sections into a message to GPT
  * Return GPT's answer

### 2 Strategy to mitigate LLM Hallucinations:

* Standard LLM Parameters:
  * setting temperature < 0.3
  * Retreival Docs : controlled generation
    * 1. It means providing enough details and constraints in the prompt to the model 
    * 2. Setting the number of top-k retrieved documents in the range 3 to 5
* Using LLM evaluation metrics:
    * Evaluating LLM response 
      * using Langchain
      * using self-consistency metrics
        1.  SelfCheck-MQAG 
        2.  SelfCheck-BERTScore
        3.  SelfCheck-Ngram
        4.  SelfCheck-NLI based
        5.  SelfCheck-Opensource model
        6.  SelfCheck-openai - since i have used openai for generation, I tried to evaluaate the self-consistency with other methods
### (Optional Challenge)  Assess the performance of the answers generated from the chatbot, given that there are no ground truth Q&A pairs provided to you.  
* For this, I have tried to use Lngchain, QAEvalChain to evaluate the LLM Response. 

  * Manual Annotation for 3 sample queries: For the provided sample queries, I have annotated the sample responses from the extracted docs as ground truth. QAEvalChain can asseses the predicted answers against the annotated responses and returns whether the predicted answer is valid or not.
  * Assesing Openai model(LLM1) generated response as ground truth and validate the response against the LLM2(RetrievalQA with chaining) response as prediction.  QAEvalChain can asseses the predicted answers(LLM2) against the responses(from LLM1) and returns whether the predicted answer is valid or not.

## Getting Started

1. **Clone the Repository:**
   Open your terminal and run the following command to clone this repository to your local machine:
   ```shell
   git clone https://github.com/saravananpsg/healthcare_llm_chatbot.git
   ```
2. **Navigate to the Project Directory:**
   Change your current directory to the project folder:
   ```shell
   cd healthcare_llm_chatbot
   ```
3. **Create a Virtual Environment:**
   It's a good practice to work within a virtual environment to manage this project dependencies. Create a virtual environment using `venv`:
   * option (1) Python in-built virtual environment
   ```shell
   python -m venv venv
   ```
   * option (2) conda environment
   ```shell
   conda create -n <env_name> python=3.11
   ```
   * Note: I have used Langchain features for most of the features for this, it seems python 3.11 is most compatible for now.
4. **Activate the Virtual Environment:**
   On macOS or Linux:
   ```shell
   source venv/bin/activate
   ```
   ```shell
   conda activate <env_name>
   ```
   On Windows:
   ```shell
   .\venv\Scripts\activate
   ```
   You should now see the virtual environment name in your terminal prompt.
5. **Install Dependencies:**
   Use `pip` to install the required dependencies in the environment.
   ```shell
   pip install -r requirements.txt
   ```
6. (Optional preference) Installation and Setup for the OpenAI API key:
   - This step is not mandatory for running the notebook per se. To obtain an OpenAI API key, follow these instructions:
     - Sign up for an OpenAI API key at [OpenAI](https://platform.openai.com/signup).
     - Once you have an API key, create .env file using :
          ```shell
          touch .env
          ```
     - Add the API key in the .env file
     ```shell
       OPENAI_API_KEY="API KEY"
     ```
7. Run the main.py to get individual response from Retriever using GPT 3.5 model
8. Run the app.py to get continuous responses from the chat model

