The purpose of this tab is to present a user guide for our application. If you would like to learn more about the methods used, we strongly advise you to read the dedicated article [Germain, 2023](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2023.1154328/full)

### Data selection
The first step to using the application is to select the data. 
You have to enter a zip file in the column on the left. Each file in the zip should represent a univariate time series. 
The supported format are **.csv**, **.txt**, **.npy** and **.EDF**. 

To make the most of the application's functionalities, we also advise you to build a CSV file with a column named **"filename"** containing the file names in the zip files and the other columns corresponding to a classification of your choice. See the following example: 

<p align="center">
<img src= "https://raw.githubusercontent.com/Machiavelli-5040/PlethLit/main/tutorial_images/csv_example.png" width="50%" />
</p>

Once the zip file is inserted, you will be able to insert the CSV file in the left column.

### Visualization of the data
Once you have entered the data, you will be able to visualize them on the tab **"Individual representation"**:

<p align="center">
<img src= "https://raw.githubusercontent.com/Machiavelli-5040/PlethLit/main/tutorial_images/representation_example.png" width="50%" />
</p>


If you have entered a CSV file that respects the precise format above, you will also be able to search for a signal you want to display based on the classification you made on your CSV file.

<p align="center">
<img src= "https://raw.githubusercontent.com/Machiavelli-5040/PlethLit/main/tutorial_images/advanced_params_example.png" width="50%" />
</p>

Once you have clicked on the circled buttons, the advanced research parameters will be applied and only the time-series respecting the wished constraints will be displayed in the research bar.

### Parameters selection

The left-hand column is also where you can select the algorithm's parameters. Some parameters are mandatory: for example, you need to enter the sampling frequency you used to make your measurements, the sub-sampling frequency, and the number of clusters desired. Other parameters are optional, you can access them by clicking on the **Advanced Parameters** button.

<p align="center">
<img src= "https://raw.githubusercontent.com/Machiavelli-5040/PlethLit/main/tutorial_images/parameters_example.png" width="50%" />
</p>

### Visualization of the algorithm results
Once the algorithm has been executed, you'll be able to view several results, divided into three tabs: 

#### Individual representation
The selection of the time series that you want to visualize is done in the same way as explained above in the **Visualization of the data** section. Now, once the wished time series is selected you will have new representations:

**Firstly**, you will see the *Time line representation of respiratory cycle categories*, also called **bar codes**: 

<p align="center">
<img src="https://raw.githubusercontent.com/Machiavelli-5040/PlethLit/main/tutorial_images/barcode_example.png" width="50%"/>
</p>

You can also zoom in on a portion of the bar code of particular interest to you, to see what's going on in more detail.

<p align="center">
<img src="https://raw.githubusercontent.com/Machiavelli-5040/PlethLit/main/tutorial_images/barcode_zoom_example.png" width="50%"/>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/Machiavelli-5040/PlethLit/main/tutorial_images/barcode_zoomed_example.png" width="50%"/>
</p>


Note that the portions in grey represents outliers that are too far to any typical respiratory cycle, in the examples we used they often corresponds to human manipulations or the death of the mouse.

**Then**, you will see the RC map of the selected Time-Series:
*A respiratory cycle (RC) map corresponds to a heat map where rows are inspiration symbols and columns are expiration symbols.*
![](https://raw.githubusercontent.com/Machiavelli-5040/PlethLit/main/tutorial_images/RCmap_example.png)

For a better idea of what each box represents, see the **Representative respiratory cycles** tab.

#### Collective representation

To be able to see results in the tab labeled **Collective representation**, you also need to have provided the csv file corresponding to your zip archive.

You will then have access to 2 selectors like the one described above, to be able to build 2 RC maps side-by-side to be able to visualize and compare the results for 2 given subsets of the parameter labels. 

![](https://raw.githubusercontent.com/Machiavelli-5040/PlethLit/main/tutorial_images/collective_parameters_example.png)
Once the parameters **choosen** and **applied** you will have two RC maps, similars to the one in the tab **Individual representation** with the difference that this time, the map will be done based on **all** the time-series corresponding to the choosen parameters.


#### Representative respiratory cycles

To be able to see results in the tab labeled **Representative cycles**, you do not need to provide the csv file corresponding to your zip archive.

You will find in this tab a grid figure displaying the medoids (real data examples closest to the centroids of the clusters) of the respiratory cycles. As before, the letters and numbers correspond to the inhalation and exhalation clusters respectively.