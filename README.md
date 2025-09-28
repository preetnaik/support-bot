Support Bot - README
====================

Description:
------------
This Python-based support bot reads a PDF document, retrieves relevant sections,
and answers user questions using AI models:
- Hugging Face DistilBERT (for question answering)
- SentenceTransformers (for semantic search and retrieval)

It also logs all actions and feedback in a log file for debugging and analysis.

Requirements:
-------------
- Python 3.8 or higher
- Packages:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers
pip install sentence-transformers
pip install PyPDF2

Optional (for web interface with Streamlit):
pip install streamlit

Files:
------
- support_bot_agent.py : Main Python script for running the bot.
- support_bot_log.txt  : Log file generated automatically; contains bot decisions and actions.
- README.txt           : This instructions file.

Usage (Terminal):
-----------------
1. Open a terminal in the folder containing support_bot_agent.py.
2. Run the script:

    python support_bot_agent.py

3. When prompted, enter the full path to your PDF document.
4. Ask questions interactively.
5. Type 'exit' to quit the bot.

Logging:
--------
- All queries, answers, and feedback are logged in support_bot_log.txt.
- Log file is created automatically in the same folder as the script.

Notes:
------
- Ensure your PDF has selectable text; scanned PDFs without OCR may not work.
- The bot works best with properly formatted sentences in the PDF.
- For better answers, make sure your PDF content is clean and continuous.
