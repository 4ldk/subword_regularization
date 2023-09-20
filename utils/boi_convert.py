import os


def convert(path, output_path):
    with open(path, encoding="utf-8") as f:
        dataset = f.read()

    dataset = dataset.split("\n")
    dataset = dataset[2:]

    new_dataset = []
    pre_tag = ""
    for d in dataset:
        if "-DOCSTART-" in d:
            continue
        if len(d) == 0:
            new_dataset.append(d)
            pre_tag = ""

        else:
            data = d.split()
            if data[-1][0] == "I" and pre_tag in ["", "O"]:
                data[-1] = list(data[-1])
                data[-1][0] = "B"
                data[-1] = "".join(data[-1])
                d = " ".join(data)

            pre_tag = data[-1][0]
            new_dataset.append(d)

    new_dataset = "\n".join(new_dataset)
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(new_dataset)


if __name__ == "__main__":
    if not os.path.exists("./test_datasets"):
        os.makedirs("./test_datasets")
    convert("./CrossWeigh/data/conllpp_test.txt", "./test_datasets/conllcw.txt")
    convert("./acl2023_conllpp/dataset/conllpp.txt", "./test_datasets/conll2023.txt")
