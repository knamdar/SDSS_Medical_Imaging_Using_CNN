import json

file_path = r"c:\Users\ernes\OneDrive - University of Toronto\Projects, Presentations, Grants, and Papers\Pascal_SDSS_Medical_Imaging_Workshop_March2026\SDSS_Medical_Imaging_Using_CNN\3_Revised_BPoint.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "markdown":
        new_source = []
        for line in cell.get("source", []):
            if "*Note: Namdar's B Point defines a new threshold determination method along the ROC curve to balance the weights of the classes and find an optimal operating point. It identifies the juncture where weights are balanced.*" in line:
                new_source.append("*Note: Namdar's B Point defines a new threshold determination method along the ROC curve to balance the weights of the classes and find an optimal operating point. The Balance line (or B line) has the formula: y = 1 - (m/n)x, where m is the number of actual negative examples and n is the number of actual positive examples. The intersection of this line with the ROC curve is the B point.*")
            else:
                new_source.append(line)
        cell["source"] = new_source

    elif cell.get("cell_type") == "code":
        source = cell.get("source", [])
        # Check if this is the cell containing Namdar's B Point calculation
        if any("Calculate Namdar's B Point" in line for line in source):
            new_source = []
            for line in source:
                if "A commonly used approximation for Namdar's B point is the intersection" in line:
                    new_source.append("# The Balance line (or B line) has the formula: y = 1 - (m/n)x\n")
                elif "of the ROC with TPR = 1 - FPR (which is equivalent to finding the break-even" in line:
                    new_source.append("# where m is the number of actual negative examples and n is the number of actual positive examples.\n")
                elif "point where Sensitivity = Specificity)." in line:
                    new_source.append("# The intersection of this line with the ROC curve is Namdar's B point.\n")
                elif "distances_to_b_line = np.abs(tpr + fpr - 1)" in line:
                    new_source.append("m = np.sum(np.array(y_true) == 0)\n")
                    new_source.append("n = np.sum(np.array(y_true) == 1)\n")
                    new_source.append("distances_to_b_line = np.abs(tpr - (1 - (m/n) * fpr))\n")
                elif "b_line_y = 1 - b_line_x" in line:
                    new_source.append("b_line_y = 1 - (m/n) * b_line_x\n")
                elif "label='B Line (TPR = 1 - FPR)'" in line:
                    new_source.append(line.replace("label='B Line (TPR = 1 - FPR)'", "label='B Line (y = 1 - (m/n)x)'"))
                else:
                    new_source.append(line)
            cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print("Notebook updated successfully.")
