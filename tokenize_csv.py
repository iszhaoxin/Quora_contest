import spacy
import argparse
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default='dev.csv')
    parser.add_argument('-output', type=str, default='dev.tokenized')
    args = parser.parse_args()

    nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

    input_file = open(args.input)
    input_csv = csv.reader(input_file, delimiter=",", doublequote=True, lineterminator="\n", quotechar='"')
    output_file = open(args.output, mode="wt")
    output_csv = csv.writer(output_file, delimiter=",", doublequote=True, lineterminator="\n", quotechar='"')
    header = next(input_csv)
    output_csv.writerow(header)
    for row in input_csv:
        row[1] = " ".join([w.text for w in nlp(row[1])])
        output_csv.writerow(row)

    input_file.close()
    output_file.close()
    