# LLM Healthcare chatbot using Openai GPT3.5 retriever model w/ chaining:

## Project objective: 
This repository is to create a conversational chatbot that can address readers’ concerns in natural language using information from the trusted articles and in the  healthcare context.

## Proposed idea:
* Using Langchain and Openai model to retrieve top-k candidates from the online sources and pass it to Openai model to find the answer for the query


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

   Use `pip` to install the project dependencies:

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



