import numpy as np
import sys
import Clean
from part_0 import part_0
from part_1 import part_1


if __name__ == '__main__':
    np.random.seed(0)
    test_X_fn, train_X_fn, train_y0_fn, train_y1_fn = sys.argv[1], sys.argv[2],\
                                                      sys.argv[3], sys.argv[4]

    # Cleaning and separating:
    test_X_df = Clean.parse(test_X_fn)

    df_0 = Clean.parse(train_X_fn, train_y0_fn)

    y_0 = df_0["אבחנה-Location of distal metastases"]
    X_0 = df_0.drop(columns=["אבחנה-Location of distal metastases"])

    df_1 = Clean.parse(train_X_fn, train_y1_fn)
    y_1 = df_1["אבחנה-Tumor size"]
    X_1 = df_1.drop(columns=["אבחנה-Tumor size"])


    # print(X_0.columns)
    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # print(test_X_df.columns)


    # Part 0
    tree_0 = part_0.fit(X_0, y_0)
    part_0.predict_to_file(tree_0, test_X_df)
    #
    # # Part 1
    tree_1 = part_1.fit(X_1, y_1)
    part_1.predict_to_file(tree_1, test_X_df)
