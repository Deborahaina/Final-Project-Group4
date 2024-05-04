
import os
OR_PATH = os.getcwd()
PATH = OR_PATH
sep = os.path.sep

RESULTS = PATH + sep + "Deborah" + sep + "Final-Project-Group4"+ sep+ "Code" + sep+ "inception_results.txt"

acc = 0.4
prec = 0.4
rec = 0.3
f1 = 0.02
output = output = (
    f"Perfomance metrics on validation set\n"
    f"----------------------------------------\n"
    f"Accuracy: {acc:.2f}\n"
    f"Precision Change: {prec:.2f}\n"
    f"Recall: {rec:.2f}\n"
    f"F1 score: {f1:.2f}\n")

with open(RESULTS, "w") as txt_file:
    txt_file.write(output)

