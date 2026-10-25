"""Build a verified dataset from the official 'Indian Constitution.pdf'.

Why: the original constitution.json had several articles whose number/title do not match
the real Constitution (e.g. 110 'Calling Attention Motion'; real Art. 110 = Money Bills).
This script re-extracts article numbers, titles, Part names and the opening text from the
PDF, keeps your Hinglish summaries wherever the old title was correct, and writes:
    data/constitution_verified.json   (used by the app)
    data/audit_report.csv             (old vs. real title for every article)
Needs the `pdftotext` command (poppler-utils). Usage:  python build_dataset.py
"""
import csv, json, re, subprocess, difflib
from pathlib import Path

ROOT = Path(__file__).parent
PDF = ROOT / "data" / "Indian Constitution.pdf"
OLD = ROOT / "data" / "constitution_original.json"

GROUPS = {  # Part -> group used by the Naive Bayes classifier
    "I": "Union, Territory & Citizenship", "II": "Union, Territory & Citizenship",
    "III": "Fundamental Rights", "IV": "Directive Principles & Duties", "IVA": "Directive Principles & Duties",
    "V": "Union Government & Judiciary", "VI": "State Government",
    "VII": "Territories, Local Govt & Centre-State", "VIII": "Territories, Local Govt & Centre-State",
    "IX": "Territories, Local Govt & Centre-State", "IXA": "Territories, Local Govt & Centre-State",
    "IXB": "Territories, Local Govt & Centre-State", "X": "Territories, Local Govt & Centre-State",
    "XI": "Territories, Local Govt & Centre-State",
    "XII": "Finance, Trade & Services", "XIII": "Finance, Trade & Services",
    "XIV": "Finance, Trade & Services", "XIVA": "Finance, Trade & Services",
    "XV": "Elections & Special Classes", "XVI": "Elections & Special Classes",
    "XVII": "Language, Emergency & Amendment", "XVIII": "Language, Emergency & Amendment",
    "XIX": "Language, Emergency & Amendment", "XX": "Language, Emergency & Amendment",
    "XXI": "Temporary & Transitional", "XXII": "Temporary & Transitional",
}
ART = re.compile(r"^\s*(\[?)(\d+[A-Z]{0,2})\.\s+(.*)$")
PART = re.compile(r"^\s*PART\s+([IVX]+[A-Z]?)\s*$")


def pdf_text():
    return subprocess.run(["pdftotext", "-layout", str(PDF), "-"], capture_output=True, text=True, check=True).stdout


def parse_toc(lines):
    start = next(i for i, l in enumerate(lines) if l.strip() == "PREAMBLE")
    arts, part, cur, i = [], None, None, start
    while i < len(lines):
        s = lines[i].strip()
        i += 1
        if re.match(r"^(THE\s+)?(FIRST\s+)?SCHEDULES?$", s) and arts and len(arts) > 300:
            break
        if not s or s in ("ARTICLES", "THE CONSTITUTION OF INDIA") or re.match(r"^\(?[ivx]+\)?\s+Contents|^\d+\s+THE CONSTITUTION", s):
            continue
        m = PART.match(s)
        if m:
            name, j = [], i
            while j < len(lines) and lines[j].strip() and lines[j].strip() == lines[j].strip().upper() \
                    and not ART.match(lines[j]) and not lines[j].strip().startswith(("CHAPTER", "ARTICLES")):
                name.append(lines[j].strip()); j += 1
            part = (m.group(1), " ".join(name).title()); cur = None
            continue
        a = ART.match(lines[i - 1])
        if a and part and (cur is None or cur["done"]):
            cur = {"num": a.group(2), "title": a.group(3).strip(), "part": part, "omitted": bool(a.group(1)), "done": False}
            arts.append(cur)
        elif cur is not None and not cur["done"]:
            cur["title"] += " " + s
        else:
            continue
        if cur["title"].rstrip().endswith((".", ".]", "etc.", "Omitted.]")):
            cur["done"] = True
    return arts


def clean_title(t):
    t = re.sub(r"\s+", " ", t).strip().rstrip(".")
    return re.sub(r"[\[\]]", "", t).strip()


def extract_descriptions(lines, arts):
    body_start = next(i for i, l in enumerate(lines) if re.match(r"^\s*1\.\s+Name and territory of the Union\.", l) and i > 900)
    end = next((i for i, l in enumerate(lines) if "THE FIRST SCHEDULE" in l and i > body_start + 5000), len(lines))
    text = re.sub(r"\s+", " ", " ".join(lines[body_start:end]))
    cursor = 0
    for a in arts:
        words = re.findall(r"\w+", a["title"])[:3]
        pat = re.compile(r"(?<![\d.])" + re.escape(a["num"]) + r"\.\s+\W*" + r"\W*".join(map(re.escape, words)), re.I)
        m = pat.search(text, cursor)
        if not m:
            a["text"] = ""; continue
        cursor = m.end()
        dash = text.find(".—", m.end() - 5, m.end() + 400)
        body = text[dash + 2: dash + 700] if dash != -1 else text[m.end(): m.end() + 500]
        body = re.split(r"_{5,}|\s\[?\d+[A-Z]{0,2}\.\s+[A-Z][^.]{3,160}\.—", body)[0]
        body = re.sub(r"\s\d+[A-Z]{0,2}\.\s*$", "", body)
        body = re.sub(r"\d*\s*\[|\]", "", body)
        body = re.sub(r"\b(Subs|Ins|Omitted|Added|Rep)\. by [^.]*\.", "", body)
        sents = re.split(r"(?<=[.;:])\s+(?=[A-Z(])", body.strip())
        out = ""
        for s in sents:
            if len(out) + len(s) > 420 and out: break
            out += (" " if out else "") + s
        a["text"] = re.sub(r"\s+\d{1,2}\.?$", "", out.strip())
    return arts


def main():
    lines = pdf_text().splitlines()
    arts = parse_toc(lines)
    arts = [a for a in arts if not a["omitted"]]
    for a in arts:
        a["title"] = clean_title(a["title"])
    arts = extract_descriptions(lines, arts)
    old = json.load(open(OLD, encoding="utf-8"))
    out, rows, kept = {}, [], 0
    for a in arts:
        o = old.get(a["num"])
        ratio = difflib.SequenceMatcher(None, (o or {}).get("title", "").lower(), a["title"].lower()).ratio() if o else 0
        ow, nw = set(re.findall(r"[a-z]{4,}", (o or {}).get("title", "").lower())), set(re.findall(r"[a-z]{4,}", a["title"].lower()))
        jac = len(ow & nw) / max(1, len(ow | nw))
        ok = ratio >= 0.4 or jac >= 0.34
        status = "OK" if ratio >= 0.55 else ("REWORDED" if ok else ("WRONG" if o else "MISSING"))
        kept += ok
        desc = a["text"] or a["title"]
        out[a["num"]] = {"title": a["title"], "description": desc,
                         "summary": o["summary"] if ok else a["title"] + ".",
                         "part": a["part"][0], "part_name": a["part"][1], "group": GROUPS.get(a["part"][0], "Other"),
                         "summary_source": "original" if ok else "auto"}
        rows.append([a["num"], (o or {}).get("title", "(missing in old data)"), a["title"], round(ratio, 2), status])
    json.dump(out, open(ROOT / "data" / "constitution_verified.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(ROOT / "data" / "audit_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["article", "old_title", "official_title", "similarity", "status"]); w.writerows(rows)
    from collections import Counter
    print(f"{len(out)} articles extracted from PDF; old summaries kept for {kept}, regenerated for {len(out)-kept}")
    print("Audit of old data:", dict(Counter(r[-1] for r in rows)))
    print("Descriptions found:", sum(1 for a in arts if a["text"]), "/", len(arts))


if __name__ == "__main__":
    main()
