# GDSC ML Round 2 Task

This repository has been created to submit the tasks for GDSC-VIT's ML Domain Round 2 recruitments in 2025.

## Task 1 - Train a Convolution Neural Network on the CIFAR-10 dataset

Scratchpad Kaggle notebook, where I've detailed every training problem as well: [Scratchpad Notebook](https://www.kaggle.com/code/utkmal/cifar-10-using-self-trained-cnn/)

Final notebook, that I've uploaded to the competition and as main.ipynb here: [Main Notebook](https://www.kaggle.com/code/utkmal/main-cifar-10-using-self-trained-cnn)

From the main notebook, a zip file was returned as output containing the "train" and "test" folders that contained the images, the .pth file that contained the best model, and the "submission.csv" file that would get submitted.

For reference, the best_model.pth and submission.csv files have been included in the "task 1" folder.

### CNN Model Architecture (along with Input and Output dimensions):

* Input [ Dimensions: (Batch_size (32), 3, 32, 32) ]
* First Convolution Layer [ (32, 3, 32, 32) -> (32, 32, 32, 32) ]
* Second Convolution Layer [ (32, 32, 32, 32) -> (32, 64, 32, 32) ]
* First Max Pooling Layer [ (32, 64, 32, 32) -> (32, 64, 16, 16) ]
* Third Convolution Layer [ (32, 64, 16, 16) -> (32, 128, 16, 16) ]
* Second Max Pooling Layer [ (32, 128, 16, 16) -> (32, 128, 8, 8) ]
* Flatten Layer [ (32, 128, 8, 8) -> (32, 8192) ]
* First FC Layer [ (32, 8192) -> (32, 256) ]
* Second FC Layer [ (32, 256) -> (32, 128) ]
* Dropout with keep_probability = 0.3
* Output Layer [ (32, 128) -> (32, 10) ]

With this architecture, when I uploaded the "submission.csv" file to Kaggle, I was able to get a score 0.77460, which is a very good score, considering this solution would've landed me to #54 or #55 position based on the end-of-competition leaderboard standings.

![Proof of score](image.png)


## Task 2 - Implement a RAG-based LLM solution, that also has function calling

Here, Langchain was used as the LLM-development framework to implement a RAG-utilizing **ReAct LLM agent**, capable of calling different tools when the need arises.

LLM model used: Gemini 2.0 Flash

Embedding model: Google Embedding 001

Tools available for the ReAct agent:

* **Arithmetic tool**: To perform simple and complex arithmetic tools

* **Web Search tool**: Powered by Tavily's API, this gives the agent the ability to search the web for latest news and information.

* **RAG tool**: You can provide this agent an external knowledge base, and it's RAG tool functionality will allow the agent to converse with the user about those knowledge bases.
  
Other features of the ReAct agent:

* Uses the FAISS vectorstore, so your data is stored only on your computer.

* Has a memory, so it is able to have a conversation rather than just answer questions one-off.
