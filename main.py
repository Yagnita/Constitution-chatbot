"""Command-line version of the Constitution NLP chatbot.   Run:  python main.py"""
from cbot.data import load_records
from cbot.chatbot import ConstitutionBot
from cbot.preprocess import explain_pipeline
from cbot.ner import extract_entities


def show(rec, score=None):
    print(f"\nArticle {rec['id']}: {rec['title']}  [{rec['group']}]" + (f"   (score {score:.2f})" if score is not None else ""))
    print("  " + rec["description"])
    if rec["summary_text"]:
        print("  Summary:", rec["summary_text"])


def main():
    bot = ConstitutionBot(load_records())
    while True:
        print("\n===== Constitution NLP Chatbot =====\n1. Ask a question (chat)\n2. Get article by number\n"
              "3. Classify a text (Naive Bayes)\n4. NLP pipeline steps\n5. Next-word suggestions (n-gram)\n6. Exit")
        c = input("Choice: ").strip()
        if c == "1":
            q = input("Your question: ")
            a = bot.answer(q)
            print("\n" + a["message"])
            for rec, sc in a["results"]:
                show(rec, sc)
            if a.get("area"):
                print("\nPredicted area:", ", ".join(f"{n} ({p:.0%})" for n, p in a["area"]))
            if a.get("entities"):
                print("Entities:", ", ".join(f"{t} [{l}]" for t, l in a["entities"]))
        elif c == "2":
            rec = bot.get_article(input("Article number (e.g. 21A): "))
            show(rec) if rec else print("Article not found")
        elif c == "3":
            for n, p in bot.classifier.predict(input("Text: "), k=3):
                print(f"  {n}: {p:.0%}")
        elif c == "4":
            for k, v in explain_pipeline(input("Text: ")).items():
                print(f"{k:24s} {v}")
        elif c == "5":
            print(", ".join(w for w, _ in bot.lm.suggest(input("Type a few words: "))) or "(no suggestion)")
        elif c == "6":
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
