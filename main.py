import numpy as np
import sys
import Clean
import part_0
import part_1

if __name__ == '__main__':
    np.random.seed(0)
    test_X_fn, train_X_fn, train_y0_fn, train_y1_fn = sys.argv[1], sys.argv[2],\
                                                      sys.argv[3], sys.argv[4]

    df_0 = Clean.parse(train_X_fn, train_y0_fn)
    test_X_df = Clean.parse(test_X_fn)

    df_1 = Clean.parse(train_X_fn, train_y1_fn)

    y_0_train, X_0_train = df_0["אבחנה-Location of distal metastases"], \
                           df_0.drop(
        columns=["אבחנה-Location of distal metastases"])

    y_1_train, X_1_train = df_1["אבחנה-Tumor size"], df_1.drop(
        columns=["אבחנה-Tumor size"])


    # Part 0
    tree_0 = part_0.fit(y_0_train, X_0_train)
    part_0.predict_to_file(tree_0, test_X_df)

    # Part 1
    tree_1 = part_1.fit(y_1_train, X_1_train)
    part_1.predict_to_file(tree_1, test_X_df)
