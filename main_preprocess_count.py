import argparse
import os
from typing import Optional
import xml.etree.ElementTree as ElementTree

# First step to run
def clean_data(year: int, phase: str):
    """Clean a SemEval dataset by removing opinions with implicit targets. This function returns the cleaned dataset."""
    filename = f"ABSA{year % 2000}_Restaurants_{phase}.xml"

    input_path = f"data/raw/{filename}"
    output_path = f"data/processed/{filename}"

    # Always reprocess the data
    tree = ElementTree.parse(input_path)

    # Dictionary to store aspect categories and their counts
    aspect_categories = {}

    # remove implicit targets
    n_null_removed = 0
    for opinions in tree.findall(".//Opinions"):
        for opinion in opinions.findall('./Opinion[@target="NULL"]'):
            opinions.remove(opinion)
            n_null_removed += 1

    # calculate descriptive statistics for remaining opinions
    n = 0
    n_positive = 0
    n_negative = 0
    n_neutral = 0
    for opinion in tree.findall(".//Opinion"):
        n += 1

        if opinion.attrib['polarity'] == "positive":
            n_positive += 1
        elif opinion.attrib['polarity'] == "negative":
            n_negative += 1
        elif opinion.attrib['polarity'] == "neutral":
            n_neutral += 1

        # Extract aspect category
        category = opinion.attrib.get('category')
        print(f"Processing opinion with category: {category}")  # Debug statement
        if category:
            if category in aspect_categories:
                aspect_categories[category] += 1
            else:
                aspect_categories[category] = 1

    if n == 0:
        print(f"\n{filename} does not contain any opinions")
    else:
        print(f"\n{filename}")
        print(f"  Removed {n_null_removed} opinions with target NULL")
        print(f"  Total number of opinions remaining: {n}")
        print(f"  Fraction positive: {100 * n_positive / n:.3f} %")
        print(f"  Fraction negative: {100 * n_negative / n:.3f} %")
        print(f"  Fraction neutral: {100 * n_neutral / n:.3f} %")

        # Print aspect categories and their counts
        print("\nAspect categories and their counts:")
        for category, count in aspect_categories.items():
            print(f"  {category}: {count}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.write(output_path)
    print(f"Stored cleaned dataset in {output_path}")

    return tree

def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2015, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--phase", default="Train", help="The phase of the dataset (Train or Test)")
    args = parser.parse_args()

    year: int = args.year
    phase: str = args.phase

    # Only clean data and print aspect categories and their counts
    clean_data(year, phase)

if __name__ == "__main__":
    main()

####delete from here
def inspect_specific_category(year: int, phase: str, category_to_inspect: str):
    """Inspect and print all opinions with a specific category."""
    filename = f"ABSA{year % 2000}_Restaurants_{phase}.xml"

    input_path = f"data/raw/{filename}"

    tree = ElementTree.parse(input_path)

    print(f"\nOpinions with category '{category_to_inspect}':")

    for sentence in tree.findall(".//sentence"):
        for opinion in sentence.findall(".//Opinion"):
            category = opinion.attrib.get('category')
            if category == category_to_inspect:
                print(ElementTree.tostring(sentence, encoding='unicode'))
                print(f"Opinion: {opinion.attrib}")
                print(f"Text: {sentence.find('text').text}")
                print()

def main():
    # Parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2015, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--phase", default="Test", help="The phase of the dataset (Train or Test)")
    parser.add_argument("--inspect", default= "FOOD#GENERAL", type=str, help="The aspect category to inspect")
    args = parser.parse_args()

    year: int = args.year
    phase: str = args.phase
    category_to_inspect: str = args.inspect

    # Clean data and print aspect categories and their counts
    clean_data(year, phase)

    if category_to_inspect:
        inspect_specific_category(year, phase, category_to_inspect)

if __name__ == "__main__":
    main()