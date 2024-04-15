<!-- <style>
table
{
    margin: auto;
}
</style> -->

### UNet style model for Evaparation Duct Height Diagnosis
This for a model for evaparation duct height regression and is named EDH-SimUNet. Use this model, you can diagnose large-scale evaparation duct height fast and precisely. The performance is shown in the following figure.

![](imgs/comparison.png)

The pictures in first line are the physical model diagnosis results, and the second line shows the EDH-SimUNet diagnosis results. Almost same! Quantitative test indicators are shown in the table below.

| Model         | Size(↓)    | RMSE(↓)    | MAE(↓)     | MedAE(↓)   | RS(↑)      | Time(↓)     |
|:-------------:|:-------:   |:-------:   |:-------:   |:-------:   |:-------:   |:-----------:|
| EDH-SimUNet	| **9.2M**   | 0.17032	  | **0.05504**| **0.03265**| 0.99985    | **7.497306**|
| UNet          | 17.3M      |**0.13707** | 0.06328    | 0.04284    | **0.99990**| 8.836329    |
| XGBoost       |    -       | 1.91156    | 1.72198    | 1.94468    | 0.98209    | 18.75348    |
| Random Forest |    -       | 5.49788    | 4.88819    | 5.53391    | 0.77509    | 16.23335    |

Before running this project, you should have dataset (not yet upload, too big 😓)

Q: How to train a EDH-SimUNet?

A: `python scripts/sim/train_sim.py`

Q: How to test the trained EDH-SimUNet?

A: `python scripts/sim/test_sim.py`
