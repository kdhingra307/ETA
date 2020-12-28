# ETA Prediction



## Preprocessing

We have three components present as the input, (i) static gtfs data, (ii) information regarding latitude-longitude, and (iii) dynamic databases, with each database corresponding to single day trip information, which contained : 
- Information regarding the bus, time of recording, and latitude-longitude coordinates.
- Speed information, 

   it initially looked like garbage values, as the values were between 0 to 5 only, but upon further evaluation, we found that the data is correct. It is reported in m/s but rescaled down by 3.6 (which is the same factor we use to convert m/s to km/hr).

### Converting dynamic data to matrix format
We divided this process into three discrete steps ([link to the code](https://github.com/kdhingra307/ETA-prediction)):

- Converting the data from SQL format to tree format
  
    SQL database was normalized to store bus and trip information at each row; to improve space and computational cost, we stored the data in a tree format, where the root of the tree is date followed by bus ids at the first level, trip ids at the second level, and an array of all timestamps along with latitude and longitude at the third level. This allowed us to iterate over any single trip easily. 

- Association of Time stamps to stops

   At the third level of the tree, we have an array of lat-long along with time, but we still do not know when the bus is at Stop-X. We aim to assign a stop id to each element of the array.
    - Example

        if we have five entries, 
        <p align="center"><img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/9ffd8ae0253215775020b67c031a04b7.svg?invert_in_darkmode" align=middle width=341.7329124pt height=16.438356pt/></p>
        
        
        and we have the location of three stops as
        <p align="center"><img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/a5a7c4ccdb4eeb505510dc9f481a6930.svg?invert_in_darkmode" align=middle width=251.48956469999996pt height=16.438356pt/></p>
        then the output of this step would be 
        <p align="center"><img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/0407d523168042a8b84decbbae747544.svg?invert_in_darkmode" align=middle width=379.95678104999996pt height=17.031940199999998pt/></p>
        
        
        We assigned the first stop to the first two entries and no stop to the last entry based on the threshold.
   
   We have tested with different thresholds of <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/f291e95d7677c66c51246926b62050b7.svg?invert_in_darkmode" align=middle width=80.39218274999999pt height=21.18721440000001pt/>, and are currently using a threshold of 150m based on the amount of error we reduce and the amount of data we loose if we reduce the threshold.
   
   
   
- Choosing which timestamps to use for each stop

   Once we have two or more entries assigned to a given stop, it becomes important to choose which entry to assign to a given stop.

   -  Experiments
     - For a transition between stop X to stop Y, we initially assigned the last entry of stop X and the last entry of stop Y, but it leads to outliers in approximately 400 stops out of 4800. We observed these outliers with all of the trips with some stops, but with others, we observed during specifics trips only. Upon further study, we found out that these stops the destination stops, and these outliers were there because drives forgot to stop the logging.
     - For a transition between stop X to stop Y, we assigned the last entry of stop X and the first entry of stop Y, this lead to the removal of outliers. Still, we introduced another problem: we are not accounting for waiting time anymore, i.e., the amount of time the bus waits at any stop. We have computed the average waiting time for each stop during a day sampled at an interval of 10 minutes (our bucket size) and use that currently.
   - We store the output in a matrix of size <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/a47ea18fd3aac0f92068206d8d9b3b40.svg?invert_in_darkmode" align=middle width=122.47453515pt height=24.65753399999998pt/>, where 144 denotes our bucket size. If the bus is going from stop X to stop Y, and the last entry at stop X was at <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/6745e5d226778c53d4c81612871d590e.svg?invert_in_darkmode" align=middle width=76.64379525pt height=22.465723500000017pt/>, and the first entry at stop Y was at <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/2eeebfd4981b9fe35af5db16fc96973a.svg?invert_in_darkmode" align=middle width=76.64379525pt height=22.465723500000017pt/>, then we will fill the first entry of the matrix corresponding to the stop <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/a0070c759142f7a0b5723afd57c22639.svg?invert_in_darkmode" align=middle width=48.196244249999985pt height=22.465723500000017pt/> transition as 2. Also, num of nodes denotes a unique transition between any two stops as we have can two buses going from stop X to stop Y and stop X to stop Z, and we need to store information regarding both in different buckets.

- Checkpoints

   - As the data was raw, there were a couple of cases we needed to handles

      - Recorded time stamps do not follow the route it needs to

         We could not put a condition that it should visit the first stop, because in some cases the bus was starting from second or <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/86dc6deb99def4fcc70d0a78c391e619.svg?invert_in_darkmode" align=middle width=25.274089499999988pt height=22.831056599999986pt/> stop too. 

      - Timestamps corresponding to a different day

         In the initial months, we observed that the entries corresponding to the next day were present in the previous day, and we discarded those as these were mostly due to the logger logging. There were approx <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/86b74cb1505883ff7fb9d96c3bcc83ff.svg?invert_in_darkmode" align=middle width=17.29457729999999pt height=22.831056599999986pt/> entries in November and December of 2019, but none were present in the later months of 2020.

      - Time more than 30 minutes,

         We discarded any entry in which the bus took more than 30 minutes to travel between two consecutive stops, as these corresponded to outliers.

      - Same time stamps

         We observed that some of the entries in the SQL database were the same (it occurs when the timestamps and bus id is the same for more than 1 entry). So, we discarded those entries.

- Complexity and Computation

   - For each day, the size of SQL was around 3-4GB. We have not stored any of the raw data (due to the space constraints as we are currently using data of 161 days), but we have stored the output after each step, and for all the data, it takes around 5GB for 161 days.
   The computation order for a single day DB is in milliseconds for step-1 and step-3 but is in <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/79646b9c54fd0d95013fcc40974bda0b.svg?invert_in_darkmode" align=middle width=24.14389889999999pt height=21.18721440000001pt/> of seconds for step-2.



## Models

We have an input of <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/828e6eea37b7b2bde3d4da289249614f.svg?invert_in_darkmode" align=middle width=182.45568659999998pt height=24.65753399999998pt/>, and we aim to predict the output of each node at time <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/628783099380408a32610228991619a8.svg?invert_in_darkmode" align=middle width=34.24649744999999pt height=21.18721440000001pt/> given previous time stamps([link to the code](https://github.com/kdhingra307/ETA)).

- Models
  - Mean Model
    - It acts as a baseline, and given the data for <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/e7063ca93167252ee7eb714b25361398.svg?invert_in_darkmode" align=middle width=67.52289059999998pt height=24.65753399999998pt/> days, we estimate the time for day <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/33359de825e43daa97171e27f6558ae9.svg?invert_in_darkmode" align=middle width=37.38576269999999pt height=22.831056599999986pt/> as <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/7f327ecdbc854bf0c35cf6e1dd269214.svg?invert_in_darkmode" align=middle width=107.25292214999997pt height=24.65753399999998pt/>.
    - We need to use mask-mean with our data because 
      - earlier data (before August 2020) had only 4000 stops covered, but later on, approximately 5800 stops are covered.
      - data is missing on certain days for certain stops; stops coverage is not the same every day.
    - To calculate mask mean, we divide the sum at each entry by the number of times that entry was not zero.
  - Seq2Seq Model
    - It acts as a second baseline, and we restructure the complete data as a matrix of size <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/fffc33e9f389b4b3b12e2db9bf1ce5d1.svg?invert_in_darkmode" align=middle width=159.89452049999997pt height=24.65753399999998pt/> where given <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/4a0dab614eaf1e6dc58146666d67ace8.svg?invert_in_darkmode" align=middle width=20.17129784999999pt height=22.465723500000017pt/> entries model is trying to predict <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/f6fac43e354f1b2ca85658091df26df1.svg?invert_in_darkmode" align=middle width=20.17129784999999pt height=22.465723500000017pt/> entries, where <img src="https://raw.githubusercontent.com/kdhingra307/ETA/master/svgs/47fc2d80c843220f30be40f8f078adfd.svg?invert_in_darkmode" align=middle width=61.255700549999986pt height=22.465723500000017pt/> constitutes the buckets (each of 10minutes).
    - The model which we have tested is based on a combination of LSTM and MLP to model the temporal relationships between the timesteps.
  - DCRNN
    - It acts as a third baseline and external model to compare with. It learns the spatial relationship between the nodes by modeling it as a diffusion process, a generalized version of Graph Convolutional Network (GCN).
    - GCN extends convolutional networks to graph networks and thus aims to improve the performance in (i) modeling the spatial relationship between nodes and (ii) allows changes in node size, which could provide significant improvement in our model as our majority of the data has only 4k nodes while the latest coming data has 6k nodes.

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
            
            