
## Continuous glucose measurement for inpatient with type 2 diabetes

[Dae-Yeon Kim, Sung-Wan Chun, jiYoung Woo, "Continuous glucose measurement for inpatient with type 2 diabetes," IEEE Dataport, July 25, 2023.](https://ieee-dataport.org/documents/continuous-glucose-measurement-inpatient-type-2-diabetes)

- ### Blood Glucose
  
  - The data on blood glucose levels, insulin administration times, and meal times are used as multiple input values. 

  <img width="812" alt="image" src="https://github.com/user-attachments/assets/1b7bf9c1-25cb-4b87-abc8-9728e9f73d1a">

  *Table 3. Continuous blood glucose statistics by gender*

  <img src="https://github.com/user-attachments/assets/50792ebc-5039-421c-9558-3b5914c81cf4" alt="Fig4. Visualization of glucose, insulin administration, and meal time values through CGM devices" width="812"/>

  *Fig4. Visualization of glucose, insulin administration, and meal time values through CGM devices*

  ##### pre-processing
    - Preprocessing was performed by collecting data using a 35-minute input window followed by a 30-minute delay to predict future values.
  <img width="400" alt="image" src="https://github.com/user-attachments/assets/198cb942-aaff-44b6-bb19-12750782d218">


- ### Electronic Medical Record (EMR)

  - EMR Data example
  
| 나이 | BMI     | 당뇨 유병기간 | BUN  | Creatinine | CRP    | C-peptide | HbA1c | Fructosamine | Urine Albumin/Creatinine ratio | file_name |
|------|---------|---------------|------|------------|--------|-----------|-------|--------------|---------------------------------|-----------|
| 64   | 20.8117 | 20.0          | 36.7 | 2.23       | 166.22 | 1.11      | 12.8  | 0            | 0.00                            | S000.xlsx |
|    |  |          | |       |  |    ...   |   |           |                            |  |
| 69   | 27.2509 | 25.0          | 43.4 | 7.63       | 134.77 | 0.00      | 5.5   | 0            | 0.00                            | S000.xlsx |

  
  - The EMR data has not been disclosed due to security reasons.
  <img width="812" alt="image" src="https://github.com/user-attachments/assets/3da8baa5-37d9-43bc-9361-4b10b962b476">

  *Table 4. EMR data affecting blood glucose in type 2 diabetes patients*
