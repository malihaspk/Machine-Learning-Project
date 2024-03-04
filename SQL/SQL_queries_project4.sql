-- SQL Query for Project 4 --
----- Group 2 -----

-- DATA CLEARNING 1: Fixing age and gender for initial model --
-----csv used: original dataset found in Kaggle-----
-- Creating table to upload data. Age is float and bmi varchar due to outliers in the dataset to prevent upload error.
CREATE TABLE stroke (
	id int,
	gender VARCHAR(10),
	age float,
	hypertension int,
	heart_disease int,
	ever_married varchar(10),
	work_type varchar(16),
	Residence_type varchar(10),
	avg_glucose_level float,
	bmi varchar,
	smoking_status varchar(20),
	stroke int);

--DROP TABLE stroke;
SELECT * FROM stroke;

-- Rounding the ages which are in decimals to nearest higher age:
UPDATE stroke
SET age = 1
WHERE age <1;

UPDATE stroke
SET age = 2
WHERE age >1 AND age <2;

-- Dropping rows where gender value is 'other':
DELETE FROM stroke
WHERE gender = 'Other';

--csv saved as: stroke_cleaned_v1--

-- DATA CLEARNING 2 for model optimization --
----- csv used: stroke_cleaned_v1 -----

-- Deleting the column smoking_status as it has no impact on our analysis:
ALTER TABLE stroke
DROP COLUMN smoking_status;

--csv saved as: stroke_cleaned_v2--

--------- AGGREGATE QUERIES ---------
-- Number of rows in dataset
SELECT COUNT(*) FROM stroke;

-- Average age of participants
SELECT AVG(age)::NUMERIC(10,0) FROM stroke;

-- Count of male and female data rows
SELECT gender, COUNT(gender) FROM stroke GROUP BY gender;

-- Distribution of work_type within the dataset
SELECT work_type, COUNT(work_type) FROM stroke GROUP BY work_type;

-- Distribution of residence type
SELECT residence_type, COUNT(residence_type) FROM stroke GROUP BY residence_type;

-- Average avg_glucose_level for 

SELECT AVG(avg_glucose_level)::NUMERIC(10,2) FROM stroke;

-- Count of stroke vs no stroke datapoints
SELECT stroke, COUNT(stroke) FROM stroke GROUP BY stroke;






















