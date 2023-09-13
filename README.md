# genomate
project for AI house camp

files description:
prompt.txt - prompt for chatGPT
brief_question.txt - test brief input for chatGPT
images_db - folder which is used as a data base for saving images
generator.py - file where you can define different image generators. It is supposed to be running on port 8080 and be available for main.py
generator_communicator.py - it is a bridge class for main to communicate with generator
main.py - backend server. It communicates with front-end and also with generator

If you want to use OpenAI capabilities, add file ".env" in the repository with a field "OPENAI_API_KEY=..."
