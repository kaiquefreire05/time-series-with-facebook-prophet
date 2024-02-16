# Time Series with Facebook Prophet - Predicting Daily Web Page Views

Author: [Kaíque Freire dos Santos]<br>
Date: [2024/02/16]

# Description:

This notebook demonstrates using the Facebook Prophet library to predict daily views of a web page. The database used refers to daily views of the Wikipedia page about Peyton Manning.

# Installation:

1. Install the Facebook Prophet library:

```pip
!pip install fbprophet
```


# Importing libraries:

```Python
from prophet import prophet
import pandas as pd
```

# Database reading:

* The database must be in CSV format and contain two columns: date and views.
* The code below reads the database:

```Python
dataset = pd.read_csv('/content/drive/MyDrive/datasets/page_wikipedia.csv')
```
# Data pre-processing:

* Rename the columns to ds and y.
* Sort the database by ascending date.

```Python
dataset = dataset[['date', 'views']].rename(columns = {'date': 'ds', 'views': 'y'})
dataset = dataset.sort_values(by = 'ds')
```

# Model construction and predictions:

* Create a Prophet model.
* Train the model on the database.
* Generate forecasts for the next 90 days.

```Python
model = Prophet()
model.fit(dataset)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
```

# Visualization of results:

* Use the plot method to visualize predictions on a graph.
* Use the plot_components method to visualize the model components.

```Python
model.plot(forecast)
model.plot_components(forecast)
```

# Comments:

* This is a basic example of using Facebook Prophet.
* You can adjust the model parameters to get better results.
* For more information about the Facebook Prophet library, see the official documentation: https://facebook.github.io/prophet/

# Files:

This repository contains the Jupyter notebook used for analysis.

# License:

* MIT

# Contributing:

Feel free to contribute to this repository by submitting a pull request.

# Improvements to the readme:

* Fix fbprophet installation error.
* Replace comments in Portuguese with comments in English.
* Add information about the database source.
* Format the code with correct indentation.
