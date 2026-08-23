"""The contract every symbolic-regression search backend follows, so
train.py/walkforward.py can swap engines without the rest of the pipeline
(combine.py, backtest.py) caring which one produced the pool.

There's no ABC here on purpose -- gplearn's mine_pool (in src/train.py) and
PySR's mine_pool_pysr (in src/engines/pysr_engine.py) both already satisfy
this shape by construction, and forcing a shared base class across two very
differently-structured libraries would add ceremony without adding safety.
This module documents the shape; it's not imported for its types.

A mine_pool(...)-shaped function takes:
    X_train, X_test: np.ndarray[n_rows, n_features]
    y_train: np.ndarray[n_rows]          -- the fitness/search target
    dates_train: np.ndarray[n_rows]      -- for date-grouped IC fitness
    feature_cols: list[str]
and returns:
    pool_train: np.ndarray[n_train_rows, n_formulas]
    pool_test: np.ndarray[n_test_rows, n_formulas]   (n_test_rows may be 0)
    formulas: list[str]                  -- human-readable, one per pool column
    models: list[object]                 -- one per pool column, each exposing
                                             .execute(X) -> np.ndarray, so
                                             predict_today.py can score fresh
                                             live data the same way regardless
                                             of which engine mined the formula
"""
