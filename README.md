# FE507

FE507 is a simple yet very powerful, 'batteries included' intuitive package for data analysing.

## How to use?

1. import the `settings` model to configure the `data_dir` where all of your data is located. (Notice: FE507 expects all
   your data to be in `csv` format.)
   ```python
   from fe507 import settings
   settings.data_dir = "./data/"  # you csv files is stored in the directory named `data` in your current directory
   ```
2. Import base classes from the package
   ```python
   from fe507 import Data, DataSource, RateOfReturnType, TimeFrameType
  ```
3. Enjoy.
# fe507
