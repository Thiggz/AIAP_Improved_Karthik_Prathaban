from sklearn import preprocessing
import yaml

with open('config.YAML') as file:
    config = yaml.safe_load(file)

rename_dict = config['rename_dict']
num_cols = config['num_cols']
cat_cols = config['cat_cols']
irrelevant = config['irrelevant']
abs_cols = config['abs_cols']
ordinal = config['ordinal']
correct_instruction = config['correct_instruction']
bmi_calc = config['bmi_key']


class PreProcess:
    def __init__(self, df):
        """Preprocessing functions - done in specified order, with outputs examinable by user

        :param df: Dataframe input for preprocessing
        """
        self.col_renamed = self._column_rename(df, rename_dict)  # Rename incorrectly named columns
        self.col_relevant = self._column_delete(self.col_renamed, irrelevant)  # Delete irrelevant columns
        self.col_imputed = self._median_imputation(self.col_relevant, num_cols)  # Median value replacement for "na"
        self.col_abs = self._absolute(self.col_imputed,
                                      abs_cols)  # Absolute operation for incorrect numerical negatives
        self.cat_std = self._category_std(self.col_abs,
                                          correct_instruction)  # Standardizing categorical values in columns(e.g 'A/a' to 'A')
        self.feature_engd = self._feature_eng(self.cat_std, ordinal)  # Feature engineering for numerical columns
        self.encoded = self._labelencode(self.feature_engd, cat_cols)  # Encoding categorical variables to numericals

        self.preprocessed = self.encoded  # Preprocessed result takes the final change

    def _column_rename(self, df, dict):
        """Rename incorrectly named features based on user's definition

        :param df: dataframe with incorrectly named features
        :param dict: dictionary with incorrect feature names (keys) and corresponding requested replacements (values)
        :return: dataframe with correctly named features
        """
        corrected = df
        for key, value in dict.items():
            corrected = corrected.rename(columns={key: value})
        return corrected

    def _column_delete(self, df, columns):
        """Delete irrelevant features based on user's configuration

        :param df: dataframe with irrelevant and relevant features (assumption)
        :param columns: list containing names for features deemed irrelevant by the user
        :return: dataframe with columns relevant to the user
        """
        corrected = df.drop(columns, axis=1)
        return corrected

    def _median_imputation(self, df, num_cols):
        """Replace any missing (NA) values with median value of column

        :param df: dataframe with NA values in numerical columns (assumption)
        :param num_cols: list of numerical feature names
        :return: dataframe without missing values for numerical features
        """
        corrected = df
        for col in num_cols:
            corrected[col] = df[col].fillna((df[col].median()))
        return corrected

    def _absolute(self, df, abs_cols):
        """Apply absolute operator for values that are incorrectly negative (for features defined by user)

        :param df: dataframe with incorrectly negative values (assumption)
        :param abs_cols: list of features which should only have positive numerical values (user defined)
        :return: dataframe without incorrect negatives
        """
        corrected = df
        for col in abs_cols:
            corrected[col] = abs(df[col])
        return corrected

    def _category_std(self, df, correct_instruction):
        """Standardize categorical names for variables with same meaning

        :param df: dataframe with categorical variables
        :param correct_instruction:
        :return:
        """
        corrected = df
        for feature, replace in correct_instruction.items():
            for incorrect in replace:
                corrected[feature].replace({incorrect: replace[incorrect]}, inplace=True)

        return corrected

    def _feature_eng(self, df, ordinal):
        """Engineer features - BMI from height and weight, and user specified ordinal variables and add to dataframe

        :param df: Dataframe with erronous values/names corrected and irrelevant features removed
        :param ordinal: nested dictionary of instructions for creating ordinal variables - condition and maximum value
        :return: Dataframe with added ordinal variables and BMI
        """
        corrected = df
        # Segment numerical data based on user's configuration for categorical variables
        global cat_cols
        global num_cols
        for feature, divisions in ordinal.items():
            new_feature = feature + '_condition'
            base = 0
            ceil = 0
            for segment in divisions:
                if type(divisions[segment]) == int or type(divisions[segment]) == float:
                    ceil = divisions[segment]
                    corrected.loc[(corrected[feature] < ceil) & (corrected[feature] >= base), new_feature] = segment
                    base = divisions[segment]
                else:
                    corrected.loc[(corrected[feature] >= base), new_feature] = segment
                cat_cols.append(new_feature)

        # Calculate BMI based on Height and Weight
        if bmi_calc == 1:
            bmi = corrected.apply(lambda x: (x.Weight / ((x.Height / 100) ** 2)), axis=1)
            corrected['BMI'] = bmi
            num_cols.append(bmi)

        return corrected

    def _labelencode(self, df, cat_cols):
        """Encodes categorical variables to numerical values for machine learning

        :param df:dataframe with categorical variables in strings per-say
        :param cat_cols:columns with categorical variables (specified by user)
        :return:dataframe with numerical and categorical values expressed numerically
        """
        corrected = df
        le = preprocessing.LabelEncoder()
        for col in cat_cols:
            corrected[col] = le.fit_transform(corrected[col])
        return corrected
