from dotenv import load_dotenv
from flask import Flask, make_response, url_for, redirect, request, render_template, current_app, g, send_file
import requests
from werkzeug.utils import secure_filename
import os
import openai
from datetime import datetime

application = Flask(__name__)

load_dotenv()


@application.route('/', methods=['GET'])
def index():
    # print("home server side")
    return render_template('index.html')


# openai.organization = ""
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


@application.route('/generate_cover', methods=['POST'])
def generate_cover():
    cover_description = request.args.get('cover_description')
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Translate this into 1. English \n\n " + cover_description + "\n\n1.",
        temperature=0.3,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    cover_description_english = response['choices'][0]['text']
    response = openai.Image.create(
        prompt = cover_description_english,
        n=1,
        size="512x512"
    )
    # print(response)
    image_url = response['data'][0]['url']
    response = requests.get(image_url)
    savename = secure_filename(str(datetime.now()) + ".jpg")
    with open("images/" + savename, "wb") as f:
        f.write(response.content)
    return image_url


@application.route('/generate_content', methods=['POST'])
def generate_content():
    subject = request.args.get('subject')
    target_age = request.args.get('target_age')
    target_age_text = ""
    if target_age == "target_college":
        target_age_text = " لطلاب الجامعات "
    if target_age == "target_highschool":
        target_age_text = " لطلاب الثانوية "
    if target_age == "target_primary":
        target_age_text = " لطلاب الأبتدائية "

    # completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    #                                           messages=[{"role": "user",  # user , assistant (chat) , system
    #                                                      "content": " اشرح لي بالتفصيل كل شيء عن " + subject + target_age_text + " ? "
    #                                                      }]
    #                                           )
    # print(completion)

    completion = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "\u062a\u062a\u0646\u0627\u0648\u0644 \u0645\u0642\u062f\u0645\u0629 \u0641\u064a \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0621 \u0627\u0644\u062d\u064a\u0648\u064a\u0629 \u062f\u0631\u0627\u0633\u0629 \u0627\u0644\u0639\u0645\u0644\u064a\u0627\u062a \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0626\u064a\u0629 \u0627\u0644\u062a\u064a \u062a\u062d\u062f\u062b \u062f\u0627\u062e\u0644 \u0627\u0644\u0623\u0646\u0638\u0645\u0629 \u0627\u0644\u062d\u064a\u0629\u060c \u0648\u062a\u0647\u062f\u0641 \u0625\u0644\u0649 \u0641\u0647\u0645 \u0627\u0644\u062a\u0641\u0627\u0639\u0644\u0627\u062a \u0648\u0627\u0644\u0639\u0645\u0644\u064a\u0627\u062a \u0627\u0644\u062d\u064a\u0648\u064a\u0629 \u0628\u0646\u0627\u0621\u064b \u0639\u0644\u0649 \u0627\u0644\u0645\u0641\u0627\u0647\u064a\u0645 \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0626\u064a\u0629. \u0648\u064a\u0634\u0645\u0644 \u0627\u0644\u0639\u0644\u0645 \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0626\u064a \u0627\u0644\u062d\u064a\u0648\u064a \u062f\u0631\u0627\u0633\u0629 \u0627\u0644\u0623\u0646\u0638\u0645\u0629 \u0627\u0644\u062d\u064a\u0629 \u0628\u062c\u0645\u064a\u0639 \u0623\u0634\u0643\u0627\u0644\u0647\u0627\u060c \u0633\u0648\u0627\u0621 \u0643\u0627\u0646\u062a \u0628\u0639\u064a\u062f\u0629 \u062c\u062f\u064b\u0627 \u0645\u062b\u0644 \u0627\u0644\u062e\u0644\u0627\u064a\u0627 \u0627\u0644\u062d\u064a\u0629 \u0623\u0648 \u0642\u0631\u064a\u0628\u0629\u064b \u0645\u062b\u0644 \u0627\u0644\u0623\u0639\u0636\u0627\u0621 \u0648\u0627\u0644\u0623\u0646\u0633\u062c\u0629 \u0627\u0644\u062d\u064a\u0629.\n\n\u062a\u062a\u0641\u0631\u0639 \u0627\u0644\u0645\u0642\u062f\u0645\u0629 \u0641\u064a \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0621 \u0627\u0644\u062d\u064a\u0648\u064a\u0629 \u0625\u0644\u0649 \u0639\u062f\u0629 \u0641\u0631\u0648\u0639\u060c \u062a\u0634\u0645\u0644:\n- \u0627\u0644\u062f\u064a\u0646\u0627\u0645\u064a\u0643\u0627 \u0627\u0644\u062e\u0644\u0648\u064a\u0629: \u0648\u0647\u064a \u062f\u0631\u0627\u0633\u0629 \u0627\u0644\u0639\u0645\u0644\u064a\u0627\u062a \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0626\u064a\u0629 \u0627\u0644\u062a\u064a \u062a\u062d\u062f\u062b \u062f\u0627\u062e\u0644 \u0627\u0644\u062e\u0644\u0627\u064a\u0627 \u0627\u0644\u062d\u064a\u0629 \u0648\u062a\u062e\u062a\u0635 \u0628\u062f\u0631\u0627\u0633\u0629 \u0627\u0644\u062a\u062f\u0641\u0642\u0627\u062a \u0627\u0644\u062d\u0631\u0643\u064a\u0629 \u0644\u0644\u062c\u0632\u064a\u0626\u0627\u062a \u0627\u0644\u0643\u0628\u064a\u0631\u0629 \u0648\u0627\u0644\u0635\u063a\u064a\u0631\u0629 \u062f\u0627\u062e\u0644 \u0627\u0644\u062e\u0644\u0627\u064a\u0627.\n- \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0621 \u0627\u0644\u062d\u064a\u0648\u0627\u0646\u064a\u0629: \u0648\u062a\u062a\u0646\u0627\u0648\u0644 \u062f\u0631\u0627\u0633\u0629 \u0627\u0644\u0645\u0639\u0627\u0644\u062c\u0627\u062a \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0626\u064a\u0629 \u0627\u0644\u062e\u0627\u0635\u0629 \u0628\u0627\u0644\u0623\u062c\u0633\u0627\u0645 \u0627\u0644\u062d\u064a\u0629 \u0645\u062b\u0644 \u0627\u0644\u0639\u0636\u0644\u0627\u062a \u0648\u0627\u0644\u0639\u0638\u0627\u0645 \u0648\u0627\u0644\u062c\u0644\u062f \u0648\u0627\u0644\u0623\u0646\u0633\u062c\u0629 \u0627\u0644\u0623\u062e\u0631\u0649 \u0627\u0644\u0645\u062a\u0639\u0644\u0642\u0629 \u0628\u0627\u0644\u0646\u0645\u0648 \u0648\u0627\u0644\u062a\u0646\u0645\u064a\u0629 \u0648\u0627\u0644\u062a\u062d\u0648\u0644\u0627\u062a.\n- \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0621 \u0627\u0644\u0628\u064a\u0648\u0643\u064a\u0645\u064a\u0627\u0626\u064a\u0629: \u0648\u0647\u064a \u062f\u0631\u0627\u0633\u0629 \u0627\u0644\u062a\u0641\u0627\u0639\u0644\u0627\u062a \u0627\u0644\u0643\u064a\u0645\u064a\u0627\u0626\u064a\u0629 \u0627\u0644\u062a\u064a \u062a\u062d\u062f\u062b \u062f\u0627\u062e\u0644 \u0627\u0644\u062e\u0644\u0627\u064a\u0627 \u0648\u062a\u062a\u0623\u062b\u0631 \u0628\u0627\u0644\u0638\u0631\u0648\u0641 \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0626\u064a\u0629 \u0645\u0646 \u062d\u064a\u062b \u0627\u0644\u062a\u0645\u062b\u064a\u0644 \u0627\u0644\u063a\u0630\u0627\u0626\u064a \u0648\u0627\u0644\u0623\u064a\u0636.\n- \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0621 \u0627\u0644\u062d\u064a\u0648\u064a\u0629 \u0627\u0644\u0637\u0628\u064a\u0629: \u0648\u062a\u062a\u0646\u0627\u0648\u0644 \u062f\u0631\u0627\u0633\u0629 \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0621 \u0627\u0644\u062d\u064a\u0648\u064a\u0629 \u0627\u0644\u062e\u0627\u0635\u0629 \u0628\u0627\u0644\u062c\u0633\u0645 \u0627\u0644\u0628\u0634\u0631\u064a\u060c \u0648\u062a\u0633\u062a\u062e\u062f\u0645 \u0639\u0644\u0649 \u0646\u0637\u0627\u0642 \u0648\u0627\u0633\u0639 \u0641\u064a \u0627\u0644\u0637\u0628 \u0627\u0644\u062d\u062f\u064a\u062b.\n\n\u064a\u0647\u062f\u0641 \u0639\u0644\u0645 \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0621 \u0627\u0644\u062d\u064a\u0648\u064a\u0629 \u0625\u0644\u0649 \u0641\u0647\u0645 \u0627\u0644\u0645\u0641\u0627\u0647\u064a\u0645 \u0627\u0644\u0623\u0633\u0627\u0633\u064a\u0629 \u0644\u0644\u0623\u0646\u0638\u0645\u0629 \u0627\u0644\u062d\u064a\u0629 \u0648\u0639\u0645\u0644\u064a\u0627\u062a\u0647\u0627 \u0645\u0646 \u062e\u0644\u0627\u0644 \u0627\u0644\u062f\u0645\u062c \u0628\u064a\u0646 \u0627\u0644\u0645\u0641\u0627\u0647\u064a\u0645 \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0626\u064a\u0629 \u0648\u0627\u0644\u0639\u0644\u0648\u0645 \u0627\u0644\u062d\u064a\u0648\u064a\u0629\u060c \u0648\u0630\u0644\u0643 \u0644\u062a\u0637\u0648\u064a\u0631 \u0646\u0645\u0627\u0630\u062c \u0639\u0645\u0644\u064a\u0629 \u0648\u0646\u0638\u0631\u064a\u0629 \u0641\u0639\u0627\u0644\u0629 \u0644\u062a\u0648\u0636\u064a\u062d \u0627\u0644\u0638\u0648\u0627\u0647\u0631 \u0627\u0644\u062d\u064a\u0648\u064a\u0629 \u062f\u0627\u062e\u0644 \u0627\u0644\u062e\u0644\u0627\u064a\u0627 \u0627\u0644\u062d\u064a\u0629.",
                    "role": "assistant"
                }
            }
        ],
        "created": 1681380029,
        "id": "chatcmpl-74o2fvYAnko9BkHQzzyNjxNt4lNTl",
        "model": "gpt-3.5-turbo-0301",
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 767,
            "prompt_tokens": 48,
            "total_tokens": 815
        }
    }

    return str(completion["choices"][0]["message"]["content"])


@application.route('/generate_subjects_titles', methods=['POST'])
def generate_subjects_titles():

    chapter_title = request.args.get('chapter_title')
    target_age = request.args.get('target_age')
    target_age_text = ""
    if target_age == "target_college":
        target_age_text = " لطلاب الجامعات "
    if target_age == "target_highschool":
        target_age_text = " لطلاب الثانوية "
    if target_age == "target_primary":
        target_age_text = " لطلاب الأبتدائية "
    # completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    #                                           messages=[{"role": "user",  # user , assistant (chat) , system
    #                                                      "content": " ما هي عنواين الموضوعات المناسبة في كتيب عن " + chapter_title + target_age_text + " ? "
    #                                                      }]
    #                                           )
    # print(completion)

    completion = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "1. \u0645\u0641\u0647\u0648\u0645 \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0621 \u0627\u0644\u062d\u064a\u0648\u064a\u0629 \u0648\u062a\u0637\u0628\u064a\u0642\u0627\u062a\u0647\u0627 \n2. \u0645\u0642\u062f\u0645\u0629 \u0639\u0646 \u0627\u0644\u062a\u0631\u0643\u064a\u0628 \u0627\u0644\u062c\u0632\u064a\u0626\u064a \u0644\u0644\u062e\u0644\u0627\u064a\u0627 \n3. \u062a\u062d\u0644\u064a\u0644 \u0648\u062a\u0637\u0628\u064a\u0642\u0627\u062a \u0627\u0644\u0637\u0627\u0642\u0629 \u0641\u064a \u0627\u0644\u062e\u0644\u0627\u064a\u0627 \n4. \u0627\u0644\u062d\u0631\u0643\u0629 \u0648\u0627\u0644\u062f\u064a\u0646\u0627\u0645\u064a\u0643\u0627 \u0641\u064a \u0627\u0644\u0623\u0646\u0638\u0645\u0629 \u0627\u0644\u062d\u064a\u0648\u064a\u0629 \n5. \u0627\u0644\u0627\u0636\u0637\u0631\u0627\u0628\u0627\u062a \u0648\u0627\u0644\u0639\u0644\u0627\u062c\u0627\u062a \u0627\u0644\u0645\u0633\u0627\u0639\u062f\u0629 \u0644\u0644\u0641\u064a\u0632\u064a\u0627\u0621 \u0627\u0644\u062d\u064a\u0648\u064a\u0629 \n6. \u062a\u0623\u062b\u064a\u0631 \u0627\u0644\u0645\u0648\u062c\u0627\u062a \u0639\u0644\u0649 \u0627\u0644\u0623\u0646\u0638\u0645\u0629 \u0627\u0644\u062d\u064a\u0648\u064a\u0629 \n7. \u0642\u0648\u0649 \u0627\u0644\u062c\u0627\u0630\u0628\u064a\u0629 \u0648\u062a\u0623\u062b\u064a\u0631\u0627\u062a\u0647\u0627 \u0639\u0644\u0649 \u0627\u0644\u062c\u0633\u0645 \u0627\u0644\u062d\u064a \n8. \u0622\u0644\u064a\u0627\u062a \u0627\u0644\u062a\u0643\u0627\u062b\u0631 \u0648\u0627\u0644\u0646\u0645\u0648 \u0641\u064a \u0627\u0644\u0623\u0646\u0638\u0645\u0629 \u0627\u0644\u062d\u064a\u0648\u064a\u0629 \n9. \u0627\u0644\u0623\u0646\u0638\u0645\u0629 \u0627\u0644\u062d\u064a\u0648\u064a\u0629 \u0648\u0639\u0644\u0627\u0642\u062a\u0647\u0627 \u0628\u0627\u0644\u0628\u064a\u0626\u0629 \u0627\u0644\u062e\u0627\u0631\u062c\u064a\u0629.",
                    "role": "assistant"
                }
            }
        ],
        "created": 1681222484,
        "id": "chatcmpl-7493cuNlfD4lFnYIVM2jPbQgObxq5",
        "model": "gpt-3.5-turbo-0301",
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 259,
            "prompt_tokens": 56,
            "total_tokens": 315
        }
    }

    return str(completion["choices"][0]["message"]["content"])


@application.route('/generate_chapters_names', methods=['POST'])
def generate_chapters_names():
    # print("generate_chapters_names server side")

    book_title = request.args.get('book_title')
    target_age = request.args.get('target_age')
    target_age_text = ""
    if target_age == "target_college":
        target_age_text = " لطلاب الجامعات "
    if target_age == "target_highschool":
        target_age_text = " لطلاب الثانوية "
    if target_age == "target_primary":
        target_age_text = " لطلاب الأبتدائية "
    # completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    #                                           messages=[{"role": "user",  # user , assistant (chat) , system
    #                                                      "content": " ما هي عنواين الفصول المناسبة لكتاب بعنوان " + book_title + target_age_text + " ? "
    #                                                      }]
    #                                           )

    completion = {"choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "1. \u0645\u0642\u062f\u0645\u0629 \u0641\u064a \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0621 \u0627\u0644\u062d\u064a\u0648\u064a\u0629\n2. \u0627\u0644\u0647\u064a\u0643\u0644 \u0648\u0627\u0644\u0648\u0638\u064a\u0641\u0629 \u0627\u0644\u062e\u0644\u0648\u064a\u064a\u0646\n3. \u0627\u0644\u062d\u0631\u0643\u0629 \u0648\u0627\u0644\u062d\u0631\u0643\u0627\u062a \u0627\u0644\u062e\u0644\u0648\u064a\u0629\n4. \u0627\u0644\u0625\u0634\u0639\u0627\u0639 \u0648\u0627\u0644\u062a\u0641\u0627\u0639\u0644\u0627\u062a \u0627\u0644\u062d\u064a\u0648\u064a\u0629\n5. \u0623\u0633\u0633 \u0627\u0644\u062d\u064a\u0627\u0629 \u0648\u0627\u0644\u062a\u0646\u0638\u064a\u0645 \u0627\u0644\u062d\u064a\u0648\u064a\n6. \u0627\u0644\u0637\u0627\u0642\u0629 \u0648\u0627\u0644\u0623\u064a\u0636 \u0627\u0644\u062d\u064a\u0648\u064a\n7. \u0627\u0644\u062a\u0648\u0627\u0632\u0646 \u0648\u0627\u0644\u062a\u0646\u0638\u064a\u0645 \u0641\u064a \u0627\u0644\u0646\u0638\u0627\u0645 \u0627\u0644\u062d\u064a\u0648\u064a\n8. \u0627\u0644\u0639\u0648\u0627\u0645\u0644 \u0627\u0644\u0628\u064a\u0626\u064a\u0629 \u0648\u0627\u0644\u062a\u0623\u062b\u064a\u0631 \u0639\u0644\u0649 \u0627\u0644\u0646\u0638\u0627\u0645 \u0627\u0644\u062d\u064a\u0648\u064a\n9. \u062a\u0637\u0628\u064a\u0642\u0627\u062a \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0621 \u0627\u0644\u062d\u064a\u0648\u064a\u0629 \u0641\u064a \u0627\u0644\u0637\u0628 \u0648\u0627\u0644\u0635\u0646\u0627\u0639\u0629\n10. \u0645\u0633\u062a\u0642\u0628\u0644 \u0627\u0644\u0641\u064a\u0632\u064a\u0627\u0621 \u0627\u0644\u062d\u064a\u0648\u064a\u0629 \u0648\u0627\u0644\u0628\u062d\u0648\u062b \u0627\u0644\u062d\u0627\u0644\u064a\u0629.",
                "role": "assistant"
            }
        }
    ],
        "created": 1681045165,
        "id": "chatcmpl-73Ovd7t3bwHoXSmSGRyiJkM1KkA2j",
        "model": "gpt-3.5-turbo-0301",
        "object": "chat.completion",
        "usage": {
        "completion_tokens": 240,
        "prompt_tokens": 49,
        "total_tokens": 289
    }
    }
    # print(type(completion))
    # print(completion)
    # print("choices[0]", completion["choices"][0]["message"]["content"])
    return str(completion["choices"][0]["message"]["content"])


if __name__ == '__main__':
    application.run(debug=True, host="0.0.0.0", use_reloader=False, port=5000)
