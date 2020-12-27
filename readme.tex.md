# ETA Prediction



## Preprocessing

We have three components present as the input, (i) static gtfs data, (ii) information regarding latitude-longitude, and (iii) dynamic databases, with each database corresponding to single day trip information, which contained : 
- Information regarding the bus, time of recording, and latitude-longitude coordinates.
- Speed information, 

   it initially looked like garbage values, as the values were between 0 to 5 only, but upon further evaluation, we found that the data is correct, and it is reported in m/s but rescaled down by a factor of 3.6 (which is the same factor we use to convert m/s to km/hr).

### Converting dynamic data to matrix format
We divided this process into three discrete steps ([link to the code](https://github.com/kdhingra307/ETA-prediction)):

- Converting the data from SQL format to tree format
  
    SQL database was normalized to store bus and trip information at each row, in order to improve space and computational cost we stored the data in a tree format, where the root of the tree is date followed by bus ids at the first level, trip ids at the second level, and an array of all timestamps along with latitude and longitude at the third level. This allowed us to easily iterate over any single trip.

- Association of Time stamps to stops

   At the third level of the tree, we have an array of lat-long along with time, but we still do not know at what time the bus is at Stop-X. We aim to assign a stop id to each element of the array.
    - Example

        if we have five entries, 
        $$
        [loc=0, loc=0.4, loc=0.7, loc=1.9, loc=2.4]
        $$
        
        
        and we have location of three three stops as
        $$
        [stop_1=0.2, stop_2=0.9, stop_3=2]\\
        $$
        then the output of this step would be 
        $$
        [loc_1=0, loc_1=0.4, loc_2=0.7, loc_3=0.9, loc_\phi=2.4]
        $$
        
        
        we assigned first stop to the first two entries, and no stop to the last entry based on threshold.
   
   We have tested with different thresholds of $50\ to\ 200m$, and are currently using threshold of 150m based on the amount of error we reduce and amount of data we loose if we reduce the threshold.
   
   
   
- Choosing which time stamps to use for each stop

   Once we have two or more entries assignmed to a given stop, it becomes important to choose which entry to assign to a given stop.

   -  Experiments
     - For a transition between stop X to stop Y, we initially assigned last entry of stop X and last entry of stop Y, but it lead to outliers in approximately 400 stops out of 4800. With some stops, we observed these outliers with all of the trips but with other we observed during specifics trips only. Upon further study, we found out that these stops the destination stops, and these outliers were there because drives forgot to stop the logging.
     - For a transition between stop X to stop Y, we assigned last entry of stop X and first entry of stop Y, this lead to removal of outliers but introduced another problem that we are not accounting for waiting time any more, i.e. the amount of time bus waits at any stop. We have computed average waiting time for each stop during a day sampled at interval of 10 minutes (our bucket), and use that currently.
   - We store the output in matrix of size $[num\_nodes, 144]$, where 144 denotes our bucket size. If bus is going from stop X to stop Y, and the last entry at stop X was at $12:14AM$, and first entry at stop Y was at $12:16AM$ then we will fill the first entry of the matrix corresponding to the stop $X-Y$ transition as 2. Also, num of nodes denotes unique transition between any two stops as we have can two buses going from stop X to stop Y, and stop X to stop Z, and we need to store information regarding both in different buckets.

- Checkpoints

   - As the data was raw, there were couple of cases we needed to handles

      - Recorded time stamps does not follow the route it needs to

         We could not put a condition that it should visit the first stop, because in some cases the bus was starting from second or $nth $ stop too. 

      - Time stamps corresponding to different day

         In the data of initial months, we observed that the entries corresponding to next day were present in the previous day, and we discarded those as these were mostly due to the logger logging. There were approx $3k $ entries in november, and december of 2019 but none were present in the later months of 2020.

      - Time more than 30 minutes,

         We discarded any entry in which bus took more that 30 minutes to travel between two consecutive stops, as these corresponded to outliers.



## Models

We have input of $[num\_nodes, time\_stamp]$, and our aim is to predict the output of each node at time $t+1$ given previous time stamps.

- Mean Model
  - It acts as a baseline, and given the data for $[1, 2, ....k]$ days, we estimate the time for day $k+1$ as $mean(1, 2, ...k)$.
  - Results are present at the 



- Training Configuration
    - Smaller Dataset
        - Data Used
            - Training and Validation
            - Jul, August, September (2020)
        - Results
          - Validation
            | Model | MAE    | MSE   | MAPE  |
            | ----- | ------ | ----- | ----- |
            | Mean  | 0.4036 | 0.989 | 0.308 |
            | LSTM  | 0.3689 | 0.71  | 0.299 |
            | DCRNN | 0.282  | 0.697 | 0.203 |

    - Bigger Dataset

        - Data Used

          - Training, and Validation
              - October, November (2019)
              - Jan, Febuarary (2020)
              - July, August (2020)
          - Testing
              - September (2020)

        - Results

            - Training
               | Model | MAE   | MSE    | RMSE   | MAPE   |
               | ----- | ----- | ------ | ------ | ------ |
               | Mean  | 0.451 | 0.889  | 0.943  | 0.545  |
               | LSTM  | 0.327 | 0.3885 | 0.6203 | 0.3772 |

            - Validation
               | Model | MAE    | MSE    | RMSE   | MAPE   |
               | ----- | ------ | ------ | ------ | ------ |
               | Mean  | -      | -      | -      | -      |
               | LSTM  | 0.328  | 0.3888 | 0.6223 | 0.3943 |

            - Testing
               | Model | MAE    | MSE    | RMSE   | MAPE   |
               | ----- | ------ | ------ | ------ | ------ |
               | Mean  | 0.436  | 1.05   | 1.025  | 0.325  |
               | LSTM  | 0.3278 | 0.3868 | 0.6201 | 0.3801 |