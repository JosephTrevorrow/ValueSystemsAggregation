## Results Information
This results folder contains dated actions/preference .csv files for each experiment. Each experiment comes as a pair of files.
- Every file here has been run on the ess data, from the European Social Survey. The specifc data can be found in ~/data/ess_example_data

Modifying the way that the factor is generated, such that instead of multiplying each action judgement, it also changes the value of action by increasing the distance between each action judgment by a factor gives more accurate results

##### 02-01-2025 Data (`factored_single_example`)
For a preference factor of 2.5 and an action factor of 5.0 gives:
- transition point of 1.40
- HCVA of 2.1 

##### Non-Factored ESS data (12-01-2025) (`single_example`)
HCVA and Transition calculation completed twice to double check 
For the original data with no factoring, the following results are observed:
- Transition Point 1.3
- HCVA Point 2.0 (cut point=2.4)

13-01-2025-against-principles.csv gives HCVA point of 1.8

#### Test Cases HCVAs (15/01/2025)
- Bottom Quartile (bottom 25%): 1.3 (cut point=1.8)
- Top Quartile (Top 25%): 2.5 (cut point=10.0)
- Extreme Egal: 3.4 (cut point=10.0)
- Extreme Util: 1.1 (cut point=1.2)

###### Top 75% of each
- General support : 4.1 (cut point 10)
- General opposition 5.5 (cut point 10)

##### Top 50% of each
- General Support: 1.1 (1.2 Cut Point) (50-pc)
- General Opposition: 6.0 (10 cut point) (50-pc)

##### ESS Data
- HCVA: 2.5 (cut points=3.7)
