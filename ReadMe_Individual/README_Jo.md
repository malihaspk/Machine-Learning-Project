## Models and Optimisation
We noticed that by running Features Importance on the original dataset whilst we were testing out different models, smoking status was not really useful. This column was therefore dropped.There was also 1 row where the gender was reported as "other", so we dropped this row. This was the first part of our optimisation, done using SQL.

During optimisation, we used various techniques to optimise for precision and recall. The techniques alongside their classification report, with explanation, is as follows:

1. ### Oversampling data and using RandomForestClassifier


![cr1](Classification-Reports/cr1.png)

    We originally went with this model to optimise because it had the highest accuracy score, despite the recall and precision being very poor for those who had been predicted to have a stroke and actually had a stroke. Majority of our optimisation was focused here, thinking we could somehow improve the model using various different algorithms.

    Thus, the first thing we tried was to oversample and scale the data. This was done using RandomOverSampler and Standard Scaler.

    Result: The false negative is 67, this is a high risk group. They have been incorrectly identified as 'no-stroke' when in reality the result is 'stroke'. This has resulted in poor precision and recall for 'stroke' prediction, so this needs to be improved. We can improve this but first we need to re-look at what features in the data is most important when it comes to training the algorithm again.

    The features most important were 'age', 'avg_glucose_level' and 'bmi'.


2. ### Oversampling data, dropping columns found less important, and using RandomForestClassifier


![cr2](Classification-Reports/cr2.png)

    This has improved the precision and recall, but only slightly, not good enough for our dataset. 

3. ### Random Forest Classifier optimised using GridSearchCV


![cr3](Classification-Reports/cr3.png)

    GridSearchCV Was used as an automated way to generate the best parameters to improve the recall. Max_features = 11 and max_depth = 14 seemed to be the best.
    Thus, the RandomForestClassifier model was re-run on the data (not oversampled) with the new params to see if it improved the recall.
    Unfortunately, precision improved with this dataset but the recall did not. n-estimators = 100 was also changed, which didn't do anything useful, so we returned this param to n-estimators = 100.
    Thus, using GridSearchCV was unsuccessful.

4. ### Outliers removed from original dataset, oversampled, using RandomForestClassifier


![cr4](Classification-Reports/cr4.png)

    This has improved the precision and recall even more, but again only slightly, not good enough for our dataset.

5. ### Balancing no-stroke to match stroke, using RandomForestClassifier


![cr5](Classification-Reports/cr5.png)

    This optimisation was the most difficult to justify. The original dataset had to be changed to balance the 'no-stroke' to match the 'stroke' numbers. This made them even. However, this meant we lost a lot of data, thousands of rows. Furthermore, any further optimisation we applied did not add value as the balanced dataset did not match or reflect the structure of the original dataset. Although the confusion matrix was fantastic and much more what we were looking for, we had to abandon this approach as in real life, it did not reflect.

    For example, in real life, the number of people that do not have strokes far outweighs the number of people that have strokes. In our balanced dataset however, the algorithm is trained to think 50:50 stroke vs no stroke. This is obviously a big problem if a clinician was to apply this algorithm to real-life people.

6. ### Outliers removed, oversampled, using Support Vector Model


![cr6](Classification-Reports/cr6.png)

    Another model that showed good potential was the Support Vector model. This was our chosen model to use, and had we more time and could potentially try other optimisations on them, we would've focused our attention here.

    This model, although giving us an accuracy of 73%, gave us a recall of 82%. Unfortunately though, the precision for prediction stroke was still only 12%. The false negatives, i.e. the 291 people it incorrectly classified as "stroke" but really they had "no-stroke", is just poor prediction when a clinician has to use it on real-life people. Unnecessary health scares, can make people worried. It also can make a clinician look incompetent at supporting people to manage their risk of strokes. We definitely do not want our model to have such a poor precision for this reason.

7. ### Outliers removed, oversampled, using Decision Tree Model


![cr7](Classification-Reports/cr7.png)

Finally, this was also tried, however it's obvious again that although the accuracy was high, precision and recall still aren't high-enough for us to reliably say this model works. This would be the second model we would work on, in order to further optimise it if we had more time.