import numpy as np
import sys
import Clean
import part_0
import part_1

if __name__ == '__main__':
    np.random.seed(0)
    test_X_fn, train_X_fn, train_y0_fn, train_y1_fn = sys.argv[1], sys.argv[2],\
                                                      sys.argv[3], sys.argv[4]

    # Part 0
    # todo make the clean work with 3 files again and not only with 1
    df_0 = Clean.parse(train_X_fn, train_y0_fn)
    test_X_df = Clean.parse(test_X_fn)

    y_0_train, X_0_train = df_0["אבחנה-Location of distal metastases"], df_0.drop(
        columns=["אבחנה-Location of distal metastases"])

    tree_0 = part_0.fit(y_0_train, X_0_train)
    part_0.predict_0(tree_0, test_X_df)

    # Part 1
    df_1 = Clean.parse(train_X_fn, train_y1_fn)

    y_1, X_1 = df_1["אבחנה-Tumor size"], df_1.drop(columns=["אבחנה-Tumor size"])
    tree_1 = part_1.fit(y_0_train, X_0_train)
    part_1.predict_1(tree_1, test_X_df)
