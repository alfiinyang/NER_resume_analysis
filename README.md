# NER_resume_analysis

This project was completed with the guide provided by Abid Ali Awan's DataCamp article on Named Entity Recognition (NER).

Here is the link to the article: https://www.datacamp.com/blog/what-is-named-entity-recognition-ner

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# **Building Resume Analysis Using Named Entity Recognition (NER)**\n",
        "\n",
        "The steps detailed in this [DataCamp article by Adib Ali Anwan](https://www.datacamp.com/blog/what-is-named-entity-recognition-ner) were used to guide ChatGPT to generate the code for building this model.\n",
        "\n",
        "\n",
        "We will create a system for analyzing resumes that helps hiring managers filter candidates based on their skills and attributes."
      ],
      "metadata": {
        "id": "JU1BPy0VvI7k"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Install and Import Libraries**\n",
        "\n",
        "We import the required packages and initialize the spaCy model and WordNet Lemmatizer for later use."
      ],
      "metadata": {
        "id": "g5QUoo-YvdNT"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install PyPDF2"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jQCYq0gK10Ss",
        "outputId": "c32a6098-7ed8-482b-ec49-a8f49992f8f7"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting PyPDF2\n",
            "  Downloading pypdf2-3.0.1-py3-none-any.whl (232 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m232.6/232.6 kB\u001b[0m \u001b[31m1.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: PyPDF2\n",
            "Successfully installed PyPDF2-3.0.1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "YYEVkOCtuJcU"
      },
      "outputs": [],
      "source": [
        "import spacy\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.stem import WordNetLemmatizer"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Convert PDF to CSV**\n",
        "\n",
        "Resumes are usually in PDF format, so we will need to convert them to a CSV file to be operated on. We can do this with PyPDF2. Here's a basic approach to convert PDF resumes into a CSV file:\n",
        "* We define a function extract_text_from_pdf that takes a PDF file path as input and returns the extracted text from the PDF.\n",
        "* Iterate over each PDF file path in the pdf_files list, extract the text from each PDF using the extract_text_from_pdf function, and store the text in a list.\n",
        "* Then create a DataFrame with columns 'ID' (to uniquely identify each resume) and 'resume_text' (to store the extracted text from resumes).\n",
        "* And lastly, we save the DataFrame to a CSV file named 'resumes.csv'."
      ],
      "metadata": {
        "id": "viICVQWq2zwZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import PyPDF2\n",
        "import pandas as pd\n",
        "\n",
        "def extract_text_from_pdf(pdf_path):\n",
        "    text = ''\n",
        "    with open(pdf_path, 'rb') as file:\n",
        "        reader = PyPDF2.PdfReader(file)\n",
        "        for page in reader.pages:\n",
        "            text += page.extract_text()\n",
        "    return text\n",
        "\n",
        "# List of PDF file paths containing resumes\n",
        "pdf_files = ['IME_INYANG_CV_.pdf',\n",
        "             'IME INYANG CV_bitnine.pdf',\n",
        "             'IME INYANG JR_CV.pdf',\n",
        "             'IME_INYANG_CV_BUA FOODS.pdf',\n",
        "             'IME_INYANG_IOM_CV (graphic design_data viz).pdf']\n",
        "\n",
        "# Extract text from each PDF resume and store it in a list\n",
        "resumes_text = [extract_text_from_pdf(pdf_path) for pdf_path in pdf_files]\n",
        "\n",
        "# Create a DataFrame with columns 'ID' and 'resume_text'\n",
        "data = pd.DataFrame({'ID': range(1, len(pdf_files)+1), 'resume_text': resumes_text})\n",
        "\n",
        "# Save the DataFrame to a CSV file\n",
        "data.to_csv('resumes.csv', index=False)"
      ],
      "metadata": {
        "id": "8vx2OXhC2xIQ"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Loading the Data and NER model**\n",
        "\n",
        "Here, the CSV file has three columns: `'ID'`, and `'resume_text'`."
      ],
      "metadata": {
        "id": "ww-zsBQmxb4l"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Load data from CSV file\n",
        "data = pd.read_csv('resumes.csv')\n",
        "\n",
        "# Load spaCy model\n",
        "nlp = spacy.load('en_core_web_sm')"
      ],
      "metadata": {
        "id": "l5WqNt7Rxgow"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Entity Ruler**\n",
        "Let's add an entity ruler pipeline to the spaCy model and create an entity ruler using a JSON file containing labels and patterns for skills:\n",
        "\n",
        "* We import EntityRuler from the `spacy.pipeline` module.\n",
        "* We import `json` module.\n",
        "* We add an entity ruler pipeline to the spaCy model using the `add_pipe` method.\n",
        "* We specify the position of the entity ruler pipeline using the `before` parameter to ensure it runs before the Named Entity Recognition (NER) pipeline.\n",
        "* We load `patterns` from a JSON file named `'skills_patterns.json'`, which contains labels and patterns for skills such as \".net\", \"cloud\", and \"aws\".\n",
        "* We convert the JSON content to a Python dictionary.\n",
        "* We add the patterns to the entity ruler using the `add_patterns` method."
      ],
      "metadata": {
        "id": "hfbCH_X7xsnJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from spacy.pipeline import EntityRuler\n",
        "# import json\n",
        "\n",
        "# Add entity ruler pipeline to spaCy model\n",
        "ruler = nlp.add_pipe(\"entity_ruler\", before=\"ner\")\n",
        "\n",
        "# Define patterns as dictionaries\n",
        "patterns = [\n",
        "    {\"label\": \"SKILL\", \"pattern\": [{\"LOWER\": \"matplotlib\"}]},\n",
        "    {\"label\": \"SKILL\", \"pattern\": [{\"LOWER\": \"python\"}]},\n",
        "    {\"label\": \"SKILL\", \"pattern\": [{\"LOWER\": \"pandas\"}]},\n",
        "    {\"label\": \"SKILL\", \"pattern\": [{\"LOWER\": \"seaborn\"}]},\n",
        "    {\"label\": \"PERSON\", \"pattern\": [{\"LOWER\": \"ime\"}, {\"LOWER\": \"okon\"}, {\"LOWER\": \"inyang\"}, {\"LOWER\": \"jnr\"}]},\n",
        "    {\"label\": \"PERSON\", \"pattern\": [{\"LOWER\": \"ime inyang\"}]},\n",
        "    {\"label\": \"PERSON\", \"pattern\": [{\"LOWER\": \"ime okon inyang\"}]}\n",
        "]\n",
        "\n",
        "# Add patterns to entity ruler\n",
        "ruler.add_patterns(patterns)"
      ],
      "metadata": {
        "id": "73LrB3Wqx5WJ"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Text Cleaning**\n",
        "Let's clean the text data using NLTK following the steps below:\n",
        "\n",
        "* We define a function `clean_text` that takes a text input and performs the cleaning.\n",
        "* We use regular expressions to remove hyperlinks, special characters, and punctuations.\n",
        "* We convert the text to lowercase and tokenize it into words.\n",
        "* We lemmatize each word to its base form using the WordNet Lemmatizer.\n",
        "* We remove English stop words using NLTK's stopwords corpus.\n",
        "* Finally, we apply this cleaning function to the 'resume_text' column in the DataFrame and store the cleaned text in a new column called 'cleaned_resume'."
      ],
      "metadata": {
        "id": "C1Lc8oxIyEhS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import nltk\n",
        "\n",
        "# Download NLTK resources\n",
        "nltk.download('punkt')  # Download the 'punkt' tokenizer resource\n",
        "nltk.download('stopwords')\n",
        "nltk.download('wordnet')\n",
        "\n",
        "import re\n",
        "from nltk.tokenize import word_tokenize\n",
        "\n",
        "# Initialize WordNet Lemmatizer\n",
        "lemmatizer = WordNetLemmatizer()\n",
        "\n",
        "def clean_text(text):\n",
        "    # Remove hyperlinks, special characters, and punctuations using regex\n",
        "    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)\n",
        "    text = re.sub(r'<.*?>', '', text)\n",
        "    text = re.sub(r'[^\\w\\s\\n]', '', text)\n",
        "\n",
        "    # Convert the text to lowercase\n",
        "    text = text.lower()\n",
        "\n",
        "    # Tokenize the text using nltk's word_tokenize\n",
        "    words = word_tokenize(text)\n",
        "\n",
        "    # Lemmatize the text to its base form for normalization\n",
        "    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]\n",
        "\n",
        "    # Remove English stop words\n",
        "    stop_words = set(stopwords.words('english'))\n",
        "    filtered_words = ' '.join([word for word in lemmatized_words if word not in stop_words])\n",
        "\n",
        "    return filtered_words\n",
        "\n",
        "# Clean the 'resume_text' column in the DataFrame\n",
        "data['cleaned_resume'] = data['resume_text'].apply(clean_text)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "k25aiPRfyg0N",
        "outputId": "858860a4-ff57-4aa6-9edc-fbae1c9aad43"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Unzipping tokenizers/punkt.zip.\n",
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Unzipping corpora/stopwords.zip.\n",
            "[nltk_data] Downloading package wordnet to /root/nltk_data...\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Entity Recognition: Visualizing Named Entities in Text with `spaCy`**\n",
        "Next:\n",
        "\n",
        "* We import the displacy module from spaCy.\n",
        "* We define options for visualization, specifying the entity labels we want to display and their corresponding colors.\n",
        "* We loop through each resume text in the DataFrame.\n",
        "* We process each resume text with the spaCy model to obtain a Doc object.\n",
        "* We use displacy.render to visualize the named entities in the text with their labels highlighted. We set `jupyter=True` to display the visualization in a Jupyter notebook.\n",
        "\n",
        "This will display the named entities for each resume text with their respective labels highlighted."
      ],
      "metadata": {
        "id": "SSizk2DFylsh"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from spacy import displacy\n",
        "\n",
        "# Define options for visualization\n",
        "options = {'ents': ['PERSON', 'GPE', 'SKILL'],\n",
        "           'colors': {'PERSON': 'orange',\n",
        "                      'GPE': 'lightgreen',\n",
        "                      'SKILL': 'lightblue'}}\n",
        "\n",
        "# Visualize named entities in each resume\n",
        "for resume_text in data['cleaned_resume']:\n",
        "    doc = nlp(resume_text)\n",
        "    displacy.render(doc, style=\"ent\", jupyter=True, options=options)\n",
        "    print('\\n\\n')"
      ],
      "metadata": {
        "id": "GTdqMKLTzAnd",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "outputId": "cb4bc639-48fe-4e2a-832b-7173902d7664"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<span class=\"tex2jax_ignore\"><div class=\"entities\" style=\"line-height: 2.5; direction: ltr\">contact alfiinyanggmailcom 234 708 005 5637 port harcourt nigeria profile portfolio datacamp medium linkedin canva skill \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " c linux centos7 graphic design canva public speaking teaching technical presentation excel power bi powerpoint etl data visualization \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    matplotlib\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " panda numpy \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    seaborn\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " workplace psychological safety cultural intelligence hypothesis based problem solving interest hobby machine learning genomics marvel comic comic explained baseball 9 ime inyang data scientist summary detail oriented organized meticulous data scientist enthusiastic team player well versed statistical analysis technique insight derivation actively seeking opportunity motivate others towards professional development eager continue growing knowledge applying skill realworld problem especially health sector education relevant coursework master data science data science sept 2024 sept 2026 university guelph canada continuing education machine learning sequential data processing analysis spatial temporal data neural network multi agent system data scientist associate certification jan 2023 present datacamp completed professional development data manipulation panda data visualization \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    matplotlib\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    seaborn\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " joining data panda \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " programming statistic \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " continuing education eda \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " supervised learning scikit learn unsupervised learning p ython machine learning tree based model \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " intro software project management aug 2022 jan 2023 alisoncom completed professional development concept characteristic software management activity involved software project management project management standard software life cycle model agile model telecommunication engineering beng nov 2011 april 2016 regent university college science technology \n",
              "<mark class=\"entity\" style=\"background: lightgreen; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    ghana\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">GPE</span>\n",
              "</mark>\n",
              " second class upper continued education c matlab work history support specialist software dev engineer may 2022 present nnpc enserv port harcourt nigeria manages linux storage system tape based seismic data processing including maintaining storage environment managing backup activity support maintains linux system including tape drive workstation server seismic data processing system network participated software field testing verify performance developed project data analyst feb 2023 present maven analytics hospital consumer assessment healthcare provider system hcahps data challenge employed \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " analyze clean extensive hcahps data reducing data redundancy 25 ensuring accurate representation developed implemented key performance indicator resulting 15 improvement identification critical healthcare metric created visually compelling dashboard report contributing 30 increase data interpretation efficiency facilitating informed decision making using seaborne umojahack africa 2023 carbon dioxide prediction challenge performed data importation cleaning analysis prediction model training using \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " contributing 270 441 placement leaderboard octave data analytics 20 data analytics project performed consumer sentiment analysis hotel data using power bi field engineer apr 2018 may 2022 nnpc enserv port harcourt nigeria installed maintained communication equipment remote field worker maintained repaired data acquisition instrument including \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    dt hp305v geophones\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " seismic data acquisition swampy terrain prepared detailed report presentation technical non technical project completion reporting mathematics teacher dec 2016 oct 2017 institute quranic study funtua nigeria provided meaningful math instruction improve math skill student junior high school level mentored student individua lly help improve math skill employed storytelling teaching technique drive student engagement improve communication student understanding resulting 23 increase student engagement achievement awarded 1 year datacamp scholarship data scientist \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    network dsn\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " taught introductory c processing geophysicist using codeblocks ide via mi crosoft team topic included c anatomy varables constant expression st atements loop contributed nnpc enserv saving approximately 427 introductory c training completed mckinsey f orward learner training communicating impact resulting increased personal profession communication effectiveness 15 tutored matlab laboratory class modelling simulation system 2015 contribut ing improvement class average 10 conducted free mathematics extramural class interested student helped increase academic performance 30 researched delivered technical presentation leveraging technology create efficency seismic acquisition nnpc enserv using engaging infographics resulting 11 employee learning commitment volunteering activity hosted antitribalism campai gns telegram name new generation nigeria national intercessory mission conducted free extramural class mathematics teacher help struggling student northern nigeria participated ndlea anti drug abuse campaign membership data scientist \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    network dsn\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " internet society isoc google developer group gdg</div></span>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "\n",
            "\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<span class=\"tex2jax_ignore\"><div class=\"entities\" style=\"line-height: 2.5; direction: ltr\">\n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    ime okon inyang jnr\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " 234 708 005 5637 alfiinyanggmailcom githubcomalfiinyang highlight qualification bachelor engineering telecommunication engineering experienced using power bi data modelling visualization experienced using \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " numpy \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    matplotlib\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " panda dat etime package data analysis machine learning preprocessing program proficiently c experienced using sql query database university level experience 6 month experience linux system storage administration seismic data processing presentation communication teaching skill practiced continuously business k nowledge sharing c training business bid problem solving workplace psychological safety practice skill gained 2022 mckinsey forward training education \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " data science jan 2023 present datacamp diploma software project management aug 2022 present national programme technology enhanced learning nptel india alisoncom bachelor engineering telecommunication engineering nov 2011 april 2016 regent university college science technology \n",
              "<mark class=\"entity\" style=\"background: lightgreen; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    ghana\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">GPE</span>\n",
              "</mark>\n",
              " second class upper project title design construction silent home security system using haptic technology 69 b work experience software development engineer support jun 2022 present nnpc enserv formerly idsl eleme petrochemica l \n",
              "<mark class=\"entity\" style=\"background: lightgreen; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    eleme\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">GPE</span>\n",
              "</mark>\n",
              " river \n",
              "<mark class=\"entity\" style=\"background: lightgreen; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    nigeria\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">GPE</span>\n",
              "</mark>\n",
              " test seismic data processing module proprietary software understudy c source code proprietary seismic data processing software manage linux storage system tape based seismic data processing including maintaining storage environment managing backup activity support maintain linux system server seismic data processing monitor report equipment health using excel spreadsheet maintain lanwlan service internet connectivity monitor equipment health report system engineer data acquisition engineer april 2018 may 2022 integrated data service ltd idsl 36 ogba road ogogugbo benin city edo nigeria utilized excel track visualize employee work progress conducted leakage smt test ground sensor geophones using voltmeter smt 300 device ascertain fit forpurpose sensor maintained repaired data acqui sition instrument including \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    dthp305v geophones\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " seismic data acquisition swampy terrain installed aintained communication equipment remote field worker researched eliver ed technical presentation alternative support process technology aid cost effective high quality data acquisition project reported head engineering mathematics teacher dec 2016 oct 2017 institute quranic study funtu formerly imam sc art secondary school prepared lecture note according outlined curriculum conducted continuous assessment test student delivered assignment corrected student assignment recorded mark reported vice principa l conducted extra moral class prepare olympiad mathematics candidate engineering intern jul 2014 sept 2014 cross river broadcasting corp crbc calabar cross river nigeria set broadcasting equipment maintained operational equipment including analog television tv radio antenna maintained electrical unit including repair replacement power socket power adapter reported engineering manager interestsachievementsawards awarded 1year datacamp free learning scholarship data scientist \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    network dsn\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " ranked 270441 zindis umojahack \n",
              "<mark class=\"entity\" style=\"background: lightgreen; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    africa 2023\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">GPE</span>\n",
              "</mark>\n",
              " carbon dioxide prediction challenge beginner taught c coworkers saving company approximately 427 beginner level training complet ed 1 data analytics project octave data analytics using power bi developed employee management system using c successfully designed constructed silent security alarm system inform deaf blind people security breach professional development member google developer group gdg 2022 present member internet society isoc 2022 present member data scientist networkdata science nigeria dsn 2021 present professional affiliation nigeria computer ciety pending training certification mckinsey forward 2022 reference available upon request</div></span>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "\n",
            "\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<span class=\"tex2jax_ignore\"><div class=\"entities\" style=\"line-height: 2.5; direction: ltr\">1 \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    ime okon inyang jnr\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " 5 ebo road airport road benin city edo nigeria 234 708 005 5637 mobile alfiinyanggmailcom email _______________________________________________________________________ personal information marital status single nationality nigerian date birth 17th december 1993 language english ibibio efik objective support transitioning oil gas related service originating developing nation first world market delivery professional experience safety consciousness whilst adapting new business system technique provide professional training immediate generation interprofessional transprofessional oil gas employee organizing modular course relatedexperienced field work experience ministry science technology uyo akwa ibom state nigeria position attachee engineering august september 2013 responsibility purchasing electrical supply assistance technical labour cross river broadcasting corp crbc calabar cross river state nigeria position attachee engineering july august 2014 responsibility operational equipment maintenance repair outsidebroadcast ob equipment setup purchasing engineering supply software usage assistance 2 imam science art secondary school funtua katsina state nigeria position nysc corper mathematics teacher december 2016 october 2017 responsibility teaching senior secondary 1 ss1 student mathematics conducting mathematics continuous assessment test final examination ss1 student representing corp member serving institution school management achievement conducted free holiday mathematics class student cowbellpedia competition ndleanysc drug free club funtua katsina state nigeria position general secretary february november 2017 responsibility record keeping every meeting event conducted organized within club preparation delivery document letter organizing meeting interest person club advancement purchase item club community development project achievement encouraged intercultural interreligious symbiosis sport game openair event coled community development project contributed sought funding project well purchased project completion item developed system increasing drugabuse awareness using digital led display nigeria christian corpers fellowship nccf funtua zone \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    katsina chapter\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " position zonal evangelism secretary february november 2017 responsibility provide spiritual moral encouragement christian corpers within funtua zone achievement helped establish new subzones within funtua zone reestablished defunct subzones 3 integrated data service ltd idsl benin edo state nigeria position instrument engineer april 2018 october 2018 responsibility maintenance recordingacquisition equipment position seismic observer junior october 2018 january 2019 responsibility line worker enrollment line worker job monitoring achievement developed swift process staff monitoring reporting using excel word position nodeman offshore january 2019 december 2019 responsibility retrieve deploy ocean bottom node obn backdeck housekeeping achievement developed toolbox checklist backdeck operation position engineer january 2020 present responsibility maintain field equipment achievement conducted knowledge sharingtraining session new recruit educational history regent university college science technolgy rucst bachelor engineering applied electronics system engineering telecomunication option second class upper 21 2016 st brians model college sbmc uyo akwa ibom state west african senior school certificate examination wassce 2010 monef kiddy international nursery primary school monef 4 first school leaving certificate fslc 2004 certificate national youth service corp certificate national service 2017 bosiet eb 2019 safety management oil gas industry caebs award national drug law enforcement agency eradication drug abuse trafficking certificate community development nov 2017 idslcnpcbgp port harcourt seismic capacity building program seismic recording certificate merit \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    jul 2018\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " shell nigeriaidslcnpcbgp rem aqarius hsse certificate appreciation jun 2019 relevant skill leadership teamwork networking presentation skill public speaking teaching workplace safety financial planning budgeting project management team motivation task analysis cultural intelligence computer literacy technical expertise digital marketing window environment apple environment microsoft word excel powerpoint omni 3d matlab pspice reference available request</div></span>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "\n",
            "\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<span class=\"tex2jax_ignore\"><div class=\"entities\" style=\"line-height: 2.5; direction: ltr\">\n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    ime inyang jr data\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " scientist software developer summary detail oriented organized meticulous data scientist enthusiastic team player well versed statistical analysis technique product development actively seeking opportunity develop skill machine learning eager continue growing knowledge applying skill real world problem especially food health industry education relevant coursework data scientist associate certification jan 2023 present datacamp received 1 year datacamp scholarship worth 200 data scientist \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    network dsn\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " completed professional development data manipulation panda data visualization \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    matplotlib\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " joining data panda \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " programming statistic \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " continuing education data visualization \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    seaborn\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " eda \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " supervised learning scikit learn unsupervised learning \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " machine learning tree based model \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " intro software project management aug 2022 jan 2023 alisoncom completed professional development concept characteristic software management activity involved software project management project management standard software life cycle model agile model telecommunication engineering beng nov 2011 april 2016 regent university college science technology \n",
              "<mark class=\"entity\" style=\"background: lightgreen; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    ghana\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">GPE</span>\n",
              "</mark>\n",
              " second class upper continued education c matla b work history software dev engineer support specialist may 2022 present nnpc enserv port harcourt nigeria contributed idea suggestion team meeting delivered update deadline design enhancement discussed issue team member provide resolution apply best practice participated software field testing verify performance developed project manages linux storage system tape based seismic data processing including maintaining storage environme nt managing backup activity support maintains linux system server seismic data processing field engineer apr 2018 may 2022 nnpc enserv port harcourt nigeria worked flexible hour across night weekend holiday shift researched delivered technical presentation alternative support process technology aid cost effective high quality data acquisition project contact alfiinyanggmailcom 234 708 005 5637 port harcourt nigeria profile \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    datacampcomprofilealfiin yang\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " linkedincominime \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    inyang jr mediumcomalfiinyang skill\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " c linux centos7 graphic design canva public speaking teaching technical presentation excel power bi powerpoint etl data visualization \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    matplotlib\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " panda numpy workplace psychological safety cultural intelligence hypothesis based problem solving hobby interest food nutrition research health research machine learning genomics marvel comic comic explained baseball 9 installed maintained communication equipment remote field worker maintained repaired data acquisition instrument including \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    dt hp305v geophones\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " seismic data acquisition swampy terrain mathematics teacher dec 2016 oct 2017 institute quranic study funtua nigeria provided meaningful math instruction improve math skill child assessed submitted class assignment determined grade reviewed work struggling student boost success chance administered assessment standardized test evaluat e student progress prepared implemented lesson plan covering required course topic mentored student individual basis help develop math skill implemented storytelling teaching technique drive student engagement improve communica tion student understanding achievement published medium article fighting covid 19 2023 using machine learning ranked top 50th percentile zindis umojahack africa 2023 carbon dioxide prediction challenge awarded 1 year datacamp schola rship data scientist \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    network dsn\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " saved company approximately 427 ginner level c training teaching coworkers c support staff completed mckinsey forward learner training program included problem solving hypothesis led approach researched delivered technical presentation leveraging technology create efficency seismic acquisition nnpc enserv completed course introduction covid 19 method detection prevention response control designed constructed silent home security system warn visually acoustically impaired people using haptic technology vibrator highest graded student matlab 94 volunteering activity host anti tribalism campai gns te legram name new generation nigeria national intercessory mission conducted free extramural class mathematics teacher help struggling student northern nigeria participated ndlea anti drug abuse campaign membership data scientist \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    network dsn\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " internet society isoc google developer group gdg</div></span>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "\n",
            "\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<span class=\"tex2jax_ignore\"><div class=\"entities\" style=\"line-height: 2.5; direction: ltr\">ime inyang data scientist graphic designer summary detail oriented organized meticulous data scientist enthusiastic team player well versed statistical analysis technique insight derivation actively seeking opportunity develop skill machine learning eager continue growing knowledge applying skill real world problem especially health sector education relevant coursework master data science data science sept 2024 sept 2026 university guelph canada continuing education machine learning sequential data processing analysis spatial temporal data neural network multi agent system data scientist associate certification jan 2023 present datacamp received 1 year datacamp scholarship worth 200 data scientist \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    network dsn\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " completed professional development data manipulation panda data visualization \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    matplotlib\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    seaborn\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " joining data panda \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " programming statistic \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " continuing educatio n eda \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " supervised learning scikit learn unsupervised learning p ython machine learning tree based model \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " intro software project management aug 2022 jan 2023 alisoncom completed professional development concept characteristic software management activity involved software project management project management standard software life cycle model agile model telecommunication engineering beng nov 2011 april 2016 regent university college science technology \n",
              "<mark class=\"entity\" style=\"background: lightgreen; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    ghana\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">GPE</span>\n",
              "</mark>\n",
              " second class upper continued education c matlab work history software dev engineer support specialist may 2022 present nnpc enserv port harcourt nigeria contributed idea suggestion team meeting delivered update deadline design enhancement discussed issue team member provide resolution apply best practice participated software field testing verify performance developed project manages linux storage system tape based seismic data processing including maintaining storage environment managing backup activity support maintains linux system server seismic data processing contact alfiinyanggmailcom 234 708 005 5637 port harcourt nigeria profile portfolio datacamp \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    canva linkedin\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " medium skill \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " c linux centos7 graphic design canva public speaking teaching technical presentation excel power bi powerpoint etl data visualization \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    matplotlib\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " panda numpy \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    seaborn\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " workplace psychological safety cultural intelligence hypothesis based problem solving interest hobby machine learning genomics marvel comic comic explained baseball 9 graphic designer digital service support freelance nov 2021 oct 2022 designed clear compelling social medium campaign online market place graphic material small business e forprofit organization including logo thumbnail digital flier poster educational humanitarian content using canva design contributed average 21 increase monthly customer engagement across 3 organization wrote clear communicating content social medium engagement resulting 20 increase conversion across 3 organization provided customer support service resolv ing payment issue 1 organization field engineer apr 2018 may 2022 nnpc enserv port harcourt nigeria designed applied engaging compelling company employee client focused infographic 2022 effectively communicate company value business perspective tech nical business non technical presentation contributing 10 increas e management satisfaction 3 increase client acquisition 12 increase employee learning commitment choice tool included canva powerpoint prepared detailed report presentation technical non technical project completion reporting installed maintained communication equipment remote field worker maintained repaired data acquisition instrument including \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    dt hp305v geophones\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " seismic data acquisition swampy terrain mathematics teacher dec 2016 oct 2017 institute quranic study funtua nigeria provided meaningful math instruction improve math skill child assessed submitted class assignment determined grade reviewed work struggling student boost success chance administered assessment standardized test evaluate student progress prepared implemented lesson plan covering required course topic mentored student individual basis help improve math skill implemented storytelling teaching technique drive student engagement improve communication student understanding achievement analyzed v isualized hospital c onsumer assessment f healthcare provider system hcahps data american hospit \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    al ociation aha\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " maven analytics using \n",
              "<mark class=\"entity\" style=\"background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    python\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">SKILL</span>\n",
              "</mark>\n",
              " aug sept 2023 provided data analysis visualization service 2 project octave dat analytics using powerbi feb apr 2023 ranked 270441 zindis umojahack \n",
              "<mark class=\"entity\" style=\"background: lightgreen; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    africa 2023\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">GPE</span>\n",
              "</mark>\n",
              " carbon dioxide prediction challenge mar 2023 awarded 1 year datacamp scholarship data scientist \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    network dsn\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " saved company approximately 427 ginner level c training teaching co worker c support staff comp leted mckinsey forward learner training program researched delivered technical presentation leveraging technology create efficency seismic acquisition nnpc enserv using engaging infographics resulting 11 employee learning commitment completed course introduction covid 19 method detection prevention response control volunteering activity host anti tribal ism campai gns telegram name new generation nigeria national intercessory mission conducted free extramural class mathematics teacher help struggling student northern nigeria participated ndlea anti drug abuse campaign membership data scientist \n",
              "<mark class=\"entity\" style=\"background: orange; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
              "    network dsn\n",
              "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
              "</mark>\n",
              " internet society isoc google developer group gdg</div></span>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Match Score**\n",
        "To match resumes with company requirements and calculate a similarity score, we can use various methods such as TF-IDF, Word Embeddings (e.g., Word2Vec, GloVe), or BERT embeddings. Here, I'll demonstrate how to calculate the similarity score using TF-IDF (Term Frequency-Inverse Document Frequency) with cosine similarity.\n",
        "\n",
        "First, let's define the requirements of the company, and then we'll calculate the similarity score for each resume based on these requirements:\n",
        "\n",
        "* We define the company requirements as a string.\n",
        "* We clean the company requirements using the clean_text function we defined earlier.\n",
        "* We calculate the TF-IDF vectors for the company requirements and each resume text.\n",
        "* We calculate the cosine similarity between the TF-IDF vector of the company requirements and each resume.\n",
        "* We sort the indices of resumes based on the similarity scores in descending order.\n",
        "* We display the top N most similar resumes along with their similarity scores.\n",
        "\n",
        "You can adjust the value of top_n to display more or fewer similar resumes. Also, you can explore other similarity calculation methods and embeddings based on your preference and requirements."
      ],
      "metadata": {
        "id": "8-a4cTsIzF0b"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "\n",
        "# Define the company requirements\n",
        "company_requirements = \"\"\"Data Analyst with experience using Python for data cleaning, data analysis, exploratory data analysis (EDA).\n",
        "                          We are also looking for someone with the ability to explain complex mathematical concepts to non-mathematicians.\"\"\"\n",
        "\n",
        "# Combine the company requirements with stopwords removed\n",
        "cleaned_company_requirements = clean_text(company_requirements)\n",
        "\n",
        "# Calculate TF-IDF vectors for the company requirements and resume texts\n",
        "tfidf_vectorizer = TfidfVectorizer()\n",
        "tfidf_matrix = tfidf_vectorizer.fit_transform(data['cleaned_resume'])\n",
        "company_tfidf = tfidf_vectorizer.transform([cleaned_company_requirements])\n",
        "\n",
        "# Calculate cosine similarity between the company requirements and each resume\n",
        "similarity_scores = cosine_similarity(company_tfidf, tfidf_matrix).flatten()\n",
        "\n",
        "# Get the indices of resumes sorted by similarity score\n",
        "sorted_indices = similarity_scores.argsort()[::-1]\n",
        "\n",
        "# Display the top 5 most similar resumes\n",
        "top_n = 5\n",
        "for i in range(top_n):\n",
        "    index = sorted_indices[i]\n",
        "    print(f\"Resume ID: {data['ID'][index]}\")\n",
        "    print(f\"Similarity Score: {similarity_scores[index]}\")\n",
        "    print()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "J9EFOOiEziKO",
        "outputId": "6c8480f5-c7c2-41fa-89a3-dcbec4feb584"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Resume ID: 1\n",
            "Similarity Score: 0.42973824001170297\n",
            "\n",
            "Resume ID: 2\n",
            "Similarity Score: 0.31343639880770646\n",
            "\n",
            "Resume ID: 5\n",
            "Similarity Score: 0.3095427060390511\n",
            "\n",
            "Resume ID: 4\n",
            "Similarity Score: 0.29207240758415465\n",
            "\n",
            "Resume ID: 3\n",
            "Similarity Score: 0.03680589209560854\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Skill Extractor Function**\n",
        "Let's create a Python function that extracts skills from a resume using the entity ruler, matches them with required skills, and generates a similarity score:\n",
        "* We define a function calculate_similarity that takes the resume text and required skills as input.\n",
        "* We process the resume text with the spaCy model.\n",
        "* We extract skills from the resume by filtering entities with the label \"SKILL\" using list comprehension.\n",
        "* We calculate the number of matching skills between the resume and required skills.\n",
        "* We calculate the similarity score by dividing the number of matching skills by the maximum of the lengths of required skills and extracted skills.\n",
        "* Finally, we return the similarity score.\n",
        "\n",
        "This function allows hiring managers to input a resume text and required skills, and it outputs a similarity score based on the matching skills. You can use this function in a loop to process multiple resumes and filter candidates based on their skills."
      ],
      "metadata": {
        "id": "MFXjF3Ww0Twx"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def calculate_similarity(resume_text, required_skills):\n",
        "    # Process the resume text with the spaCy model\n",
        "    doc = nlp(resume_text)\n",
        "\n",
        "    # Extract skills from the resume using the entity ruler\n",
        "    skills = [ent.text.lower() for ent in doc.ents if ent.label_ == \"SKILL\"]\n",
        "\n",
        "    # Calculate the number of matching skills with required skills\n",
        "    matching_skills = [skill for skill in skills if skill in required_skills]\n",
        "    num_matching_skills = len(matching_skills)\n",
        "\n",
        "    # Calculate the similarity score\n",
        "    similarity_score = num_matching_skills / max(len(required_skills), len(skills))\n",
        "\n",
        "    return similarity_score"
      ],
      "metadata": {
        "id": "REO5GSGh0nf4"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "for index, resume_text in data[['cleaned_resume']].itertuples():\n",
        "  print(f\"Resume ID: {data['ID'][index]}\")\n",
        "  required_skills = [\"matplotlib\", \"python\", \"pandas\", \"seaborn\"]\n",
        "  similarity_score = calculate_similarity(resume_text, required_skills)\n",
        "  print(\"Similarity Score:\", similarity_score)"
      ],
      "metadata": {
        "id": "drqkxwrP0vei",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ad7be88f-089d-4602-8cfd-ce31ed3e47fe"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Resume ID: 1\n",
            "Similarity Score: 1.0\n",
            "Resume ID: 2\n",
            "Similarity Score: 0.75\n",
            "Resume ID: 3\n",
            "Similarity Score: 0.0\n",
            "Resume ID: 4\n",
            "Similarity Score: 1.0\n",
            "Resume ID: 5\n",
            "Similarity Score: 1.0\n"
          ]
        }
      ]
    }
  ]
}
