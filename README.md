## Medical AI Chatbot - Powered by Groq API & Mixtral-8x7b-32768

## Overview

This project is an advanced AI-driven medical chatbot leveraging the **Groq API** with the **Mixtral-8x7b-32768** model. Additionally, the model has been fine-tuned using the **Gale Encyclopedia of Medicine**, ensuring precise and contextually rich medical responses.

## Features

- **AI-Powered Medical Assistance**: Provides insights based on trained medical literature.
- **Custom Model Training**: Enhanced understanding of medical terminologies and conditions.
- **Efficient Query Handling**: Fast response times using Groq API.
- **Scalable Deployment**: Easily deployable on AWS with CI/CD integration.
- **Data Storage & Retrieval**: Uses **Pinecone** for vector storage and fast search.

## Installation & Setup

### 1. Clone the Repository

```sh
 git clone https://github.com/YOUR-REPO-LINK.git
 cd YOUR-REPO-NAME
```

### 2. Create a Conda Environment

```sh
conda create -n medibot python=3.10 -y
conda activate medibot
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a **.env** file in the root directory and add your credentials:

```env
PINECONE_API_KEY = "your_pinecone_api_key"
GROQ_API_KEY = "your_groq_api_key"
```

### 5. Store Embeddings to Pinecone

```sh
python store_index.py
```

### 6. Run the Application

```sh
python app.py
```

Now, open `localhost:5000` to interact with the chatbot.

## Tech Stack

- **Python** (Core language)
- **Groq API** (LLM API for chatbot responses)
- **LangChain** (LLM framework)
- **Flask** (Web framework)
- **Pinecone** (Vector database for embeddings)
- **AWS CI/CD** (Cloud deployment with GitHub Actions)

## Deployment on AWS

### Step 1: AWS Configuration

1. **Login to AWS Console**
2. **Create an IAM User** with the following access policies:
   - AmazonEC2ContainerRegistryFullAccess
   - AmazonEC2FullAccess

### Step 2: Build & Push Docker Image

```sh
docker build -t medicalchatbot .
docker tag medicalchatbot:latest 970547337635.dkr.ecr.ap-south-1.amazonaws.com/medicalchatbot
docker push 970547337635.dkr.ecr.ap-south-1.amazonaws.com/medicalchatbot
```

### Step 3: Setup EC2 & Deploy

1. **Create an EC2 Instance (Ubuntu)**
2. **Install Docker on EC2**:

```sh
sudo apt-get update -y
sudo apt-get upgrade
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

3. **Pull and Run the Docker Image**:

```sh
docker pull 970547337635.dkr.ecr.ap-south-1.amazonaws.com/medicalchatbot
docker run -d -p 5000:5000 medicalchatbot
```

### Step 4: Set Up GitHub Secrets for CI/CD

In **GitHub Actions**, configure the following secrets:

- AWS\_ACCESS\_KEY\_ID
- AWS\_SECRET\_ACCESS\_KEY
- AWS\_DEFAULT\_REGION
- ECR\_REPO
- PINECONE\_API\_KEY
- GROQ\_API\_KEY

## Contributions

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

MIT License

## Contact

For queries, contact [Your Name] at [Your Email].

