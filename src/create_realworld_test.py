import os
import pandas as pd


OUTPUT = r"datasets\text\realworld_test.csv"


# =========================================================
# REAL-WORLD GENERALIZATION TEST SET
# =========================================================

samples = [

    # -----------------------------------------------------
    # HUMAN — CASUAL
    # -----------------------------------------------------

    {
        "text": "Honestly, trying to keep up with new technology is exhausting. It feels like every time I finally learn a new app, three more show up.",
        "label": 0,
        "category": "Human-Casual"
    },

    {
        "text": "Yesterday I went to college early because I had to submit my project. I thought I would finish quickly, but my friend called me for lunch and we ended up talking for almost two hours.",
        "label": 0,
        "category": "Human-Personal"
    },

    {
        "text": "I don't really like studying late at night. I can sit with the book for hours, but after a certain point nothing actually stays in my head.",
        "label": 0,
        "category": "Human-Casual"
    },

    {
        "text": "My laptop suddenly stopped working yesterday and I spent nearly an hour checking cables before realizing the charger wasn't plugged in properly.",
        "label": 0,
        "category": "Human-Personal"
    },

    # -----------------------------------------------------
    # HUMAN — STUDENT / COLLEGE
    # -----------------------------------------------------

    {
        "text": "Our professor explained the topic today, but I still didn't understand the last part. I think I'll go through the notes again before tomorrow's class.",
        "label": 0,
        "category": "Human-Student"
    },

    {
        "text": "I started working on the project a little late, so now I'm trying to finish the documentation and testing before the deadline.",
        "label": 0,
        "category": "Human-Student"
    },

    {
        "text": "The first version of our program had quite a few errors. We fixed most of them, although there are still some things that need testing.",
        "label": 0,
        "category": "Human-Technical"
    },

    # -----------------------------------------------------
    # HUMAN — OPINION
    # -----------------------------------------------------

    {
        "text": "Technology makes life easier, but I think we depend on it a little too much sometimes. There are days when I just want to put my phone away and do something without checking a screen.",
        "label": 0,
        "category": "Human-Opinion"
    },

    {
        "text": "I don't think every problem needs a complicated solution. Sometimes the simple approach works better, especially when people actually have to use the system.",
        "label": 0,
        "category": "Human-Opinion"
    },

    # -----------------------------------------------------
    # HUMAN — LONGER
    # -----------------------------------------------------

    {
        "text": "College life has been pretty unpredictable this semester. Some weeks are relaxed and I have plenty of time to work on projects, while other weeks seem to have assignments, tests and presentations all at once. I usually make a plan at the beginning of the week, but somehow something always changes. Still, I think I'm getting better at managing everything compared with last year.",
        "label": 0,
        "category": "Human-Long"
    },

    {
        "text": "When I first started learning programming, I used to get stuck on very small mistakes. I would spend a lot of time looking at the same piece of code without noticing what was wrong. Eventually I started using simple debugging techniques and reading error messages more carefully. I still make plenty of mistakes, but now I usually have a better idea of where to look.",
        "label": 0,
        "category": "Human-Long"
    },


    # =====================================================
    # AI — GENERAL
    # =====================================================

    {
        "text": "Artificial intelligence is transforming the modern world by enabling machines to analyze information, recognize patterns, and generate meaningful content. These capabilities are being applied across numerous industries to improve efficiency and support decision-making.",
        "label": 1,
        "category": "AI-General"
    },

    {
        "text": "Artificial intelligence has become one of the most transformative technologies of the modern era. It is being used across healthcare, education, finance, manufacturing, transportation, and many other industries.",
        "label": 1,
        "category": "AI-General"
    },

    {
        "text": "AI systems generate text by predicting the most likely sequence of words based on patterns learned from large datasets. This allows modern language models to produce coherent and contextually relevant responses.",
        "label": 1,
        "category": "AI-Technical"
    },

    # -----------------------------------------------------
    # AI — ACADEMIC
    # -----------------------------------------------------

    {
        "text": "The integration of artificial intelligence into educational environments has significantly enhanced the potential for personalized learning. By analyzing student performance and behavioral patterns, intelligent systems can provide adaptive instructional content and targeted feedback.",
        "label": 1,
        "category": "AI-Academic"
    },

    {
        "text": "The development of intelligent computational systems has introduced significant opportunities for improving organizational productivity. Automated data analysis, predictive modeling, and natural language processing enable institutions to make more informed decisions.",
        "label": 1,
        "category": "AI-Academic"
    },

    # -----------------------------------------------------
    # AI — FORMAL
    # -----------------------------------------------------

    {
        "text": "The implementation of advanced technological solutions continuously enhances operational efficiency while simultaneously reducing resource consumption. Organizations can therefore leverage digital transformation to achieve sustainable improvements in productivity.",
        "label": 1,
        "category": "AI-Formal"
    },

    {
        "text": "Modern technological developments continue to reshape the way individuals interact with digital systems. As computational capabilities increase, organizations are increasingly adopting intelligent solutions to optimize processes and improve overall performance.",
        "label": 1,
        "category": "AI-Formal"
    },

    # -----------------------------------------------------
    # AI — LONG
    # -----------------------------------------------------

    {
        "text": "Artificial intelligence has emerged as a significant technological development with applications across a wide range of domains. Modern AI systems are capable of processing large volumes of information, identifying complex patterns, generating natural language, and assisting users with a variety of tasks. In healthcare, these systems can support medical analysis and improve diagnostic workflows. In education, they can provide personalized learning experiences and automated feedback. Businesses can utilize AI to optimize operations, analyze customer behavior, and improve strategic decision-making. As these technologies continue to evolve, their influence on society is expected to increase considerably.",
        "label": 1,
        "category": "AI-Long"
    },

    {
        "text": "The rapid advancement of digital technologies has fundamentally changed the way organizations operate and communicate. Artificial intelligence, in particular, provides powerful capabilities for analyzing data, automating repetitive processes, and generating valuable insights. These systems can improve productivity while reducing operational costs and enabling organizations to respond more effectively to changing market conditions. However, successful implementation requires careful consideration of data quality, security, ethical responsibilities, and human oversight. Consequently, organizations must adopt balanced strategies that combine technological innovation with responsible governance.",
        "label": 1,
        "category": "AI-Long"
    },


    # =====================================================
    # AI — SHORT
    # =====================================================

    {
        "text": "Artificial intelligence improves productivity by automating repetitive tasks and analyzing complex information efficiently.",
        "label": 1,
        "category": "AI-Short"
    },

    {
        "text": "Modern AI systems can generate coherent text by learning statistical patterns from extremely large collections of data.",
        "label": 1,
        "category": "AI-Short"
    },


    # =====================================================
    # HUMAN — SHORT
    # =====================================================

    {
        "text": "I forgot my charger at home again.",
        "label": 0,
        "category": "Human-VeryShort"
    },

    {
        "text": "The bus was late this morning.",
        "label": 0,
        "category": "Human-VeryShort"
    },

    {
        "text": "I finished my assignment yesterday.",
        "label": 0,
        "category": "Human-Short"
    },

    {
        "text": "My friend helped me fix the code.",
        "label": 0,
        "category": "Human-Short"
    }
]


# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(samples)


# =========================================================
# LENGTH GROUP
# =========================================================

def length_group(text):

    count = len(text.split())

    if count <= 10:
        return "very_short"

    elif count <= 30:
        return "short"

    elif count <= 60:
        return "medium"

    return "long"


df["length_group"] = df["text"].apply(
    length_group
)


# =========================================================
# SAVE
# =========================================================

os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 60)
print("REAL-WORLD TEST DATASET CREATED")
print("=" * 60)

print()
print("Output:")
print(OUTPUT)

print()
print(
    f"Total samples: {len(df)}"
)

print()
print("LABEL DISTRIBUTION")
print("==================")

print(
    df["label"]
    .map({
        0: "Human",
        1: "AI"
    })
    .value_counts()
    .to_string()
)

print()
print("CATEGORY DISTRIBUTION")
print("=====================")

print(
    df["category"]
    .value_counts()
    .to_string()
)

print()
print("LENGTH DISTRIBUTION")
print("===================")

print(
    pd.crosstab(
        df["length_group"],
        df["label"].map({
            0: "Human",
            1: "AI"
        })
    )
    .to_string()
)

print()
print("=" * 60)
print("REAL-WORLD TEST DATASET READY")
print("=" * 60)