
# Google Drive RAG 

Allows to perform queries over documents on the google drive using LLM 


## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  pip install -r requirements.txt
```
Connect to Google Drive API: 
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;You need to create a service account folllowing the steps mentioned [here](https://cloud.google.com/iam/docs/keys-create-delete)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Get your json file and rename to credentials.json and move to the project root

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Don't forget to share the google drive folder you intend to use with the service account.

`credentials.json` should have the following format:

```JSON
{
  "type": "service_account",
  "project_id": "",
  "private_key_id": "",
  "private_key": "",
  "client_email": "",
  "client_id": "",
  "auth_uri": "",
  "token_uri": "",
  "auth_provider_x509_cert_url": "",
  "client_x509_cert_url": "",
  "universe_domain": "googleapis.com"
}
```

Create a `.env` file and add the following evironmental variable:
`OPENAI_API_KEY`
  
To run the project:
```bash
  streamlit run query.py
```


## Usage

Google Folder ID can be extracted from the drive URL.

For example, the folder_id of `https://drive.google.com/drive/folders/1w7XryYu6mL9VLmfyqUkA4_fRnDbsCqV-` is `1w7XryYu6mL9VLmfyqUkA4_fRnDbsCqV-`.
