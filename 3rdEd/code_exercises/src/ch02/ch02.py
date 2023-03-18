'''
Created on Dec 23, 2022

@author: prhodes
'''

from pathlib import Path
import pandas as pd
import numpy as np
import tarfile
import urllib.request
import matplotlib.pyplot as plt
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import  LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import joblib
from scipy import stats
from keras.engine.data_adapter import broadcast_sample_weight_modes




def main():
    
    print( "Welcome to Machine Learning!")

    pd.set_option("display.max_columns",10)

    housing = load_housing_data()
    
    print( "Head: \n" )
    print( housing.head(5) )
    
    print( "Info: \n" )
    
    housing.info()
    
    print( "Column counts: \n" )
    
    print( housing["ocean_proximity"].value_counts() )
    
    print( "Describe: \n" )
    
    print( housing.describe() )
    
    #    housing.hist(bins=50, figsize=(12,8))
    #    plt.show()
    
    # one approach but will generate different test and train sets everytime
    # this is run! Not realy what we want. One way to sort of address that is to fix
    # the random number generator seed to a fixed value before doing this
    # train_set, test_set = shuffle_and_split_data(housing, 0.2)
    # print( "Len(housing) ", len(housing), "\n" )
    # print( "Len(train_set)", len(train_set), "\n")
    # print( "Len(test_set)", len(test_set), "\n" )


    # This approach works by just using the row index as the id, but it has the shortcoming that
    # ids' aren't particularly stable. You have to ensure that any new data is only ever appended
    # to the dataset, and that no row is ever deleted. 
    housing_with_id = housing.reset_index()
    
    print( "Info: \n" )    
    housing_with_id.info()
        
    print( "Head: \n" )
    print( housing_with_id.head(5) )
    
    train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
    
    print( "Len(housing_with_id) ", len(housing_with_id), "\n" )
    print( "Len(train_set)", len(train_set), "\n")
    print( "Len(test_set)", len(test_set), "\n" )    
    
    # you can also make a synthetic identifier using the longitude and latitude since
    # those effectively don't change. Note however, in this dataset the location information
    # is fairly coarse, so you run the risk of multiple districts getting the same id.
    
    housing_with_id["id"] = housing["longitude"]*1000 + housing["latitude"]
    
    print( "Info: \n" )    
    housing_with_id.info()
        
    print( "Head: \n" )
    print( housing_with_id.head(5) )

    
    train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")

    print( "Len(housing_with_id) ", len(housing_with_id), "\n" )
    print( "Len(train_set)", len(train_set), "\n")
    print( "Len(test_set)", len(test_set), "\n" )    


    # another approach is to use an sklearn built-in like train_test_split
    # note use of random_state=42 (or any fixed value) to make sure we get the same
    # splits each time. We want this because we don't want to eventually accidentally
    # "see" all of what should be "test" data in the "training" set! This defeats the
    # purpose of splitting the data in the first place.
    train_set, test_set = train_test_split( housing, test_size=0.2, random_state=42)
    
    print( "Len(housing_with_id) ", len(housing_with_id), "\n" )
    print( "Len(train_set)", len(train_set), "\n")
    print( "Len(test_set)", len(test_set), "\n" )    
    
    
    # Everything we've done above is based on pure random sampling, which is usually good,
    # BUT if the data has separations ("strata") that need to be preserved in the distributions
    # of the test and training sets respectively, then we don't want pure random sampling.
    # Instead we want stratified sampling that first preserves the distribution by strata
    # and then randomly samples within each strata. 
    
    # In this example, the average income in a district is a particularly important attribute
    # and so we want to preserve any stratification around categories of income level.
    # so we'll construct an income_cat attribute by discretizing the median_income attribute
    # which is a continuously ranging attribute.
    
    housing["income_cat"] = pd.cut( housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
                                    labels=[1, 2, 3, 4, 5])
    
    
    # if we want to visualize the distribution of our new income_cat attribute
    # housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
    # plt.xlabel("Income category" )
    # plt.ylabel( "Number of districts" )
    # plt.show()
    
    # the somewhat advanced way to do our split using stratification. Note that the
    # split() method returns the training and test indices respectively, and not the
    # actual data
    
    # splitter = StratifiedShuffleSplit( n_splits=10, test_size=0.2, random_state=42)
    # strat_splits = []
    # for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    #    strat_train_set_n = housing.iloc[train_index]
    #    strat_test_set_n = housing.iloc[test_index]
    #    strat_splits.append([strat_train_set_n, strat_test_set_n])
    
    # and now just for convenience, just grab the first split
    # strat_train_set, strat_test_set = strat_splits[0]
    
    # NOTE: since stratified sampling is fairly common, sklearn has a built-in mechanism to make
    # this simpler. Just use the "stratify" argument with train_test_split()
    strat_train_set, strat_test_set = train_test_split( housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
    
    print( "Len(strat_train_set)", len(strat_train_set), "\n")
    print( "Len(strat_test_set)", len(strat_test_set), "\n" )    
    
    # now that we have our test and train sets established, we don't really need
    # the income_cat attribute anymore so we can drop it
    for set_ in (strat_train_set, strat_test_set):
        set_.drop( "income_cat", axis=1, inplace=True)
            
    print( "\nHead (strat_train_set): \n" )
    print( strat_train_set.head(5) )    
    
    print( "\nHead (strat_test_set): \n" )
    print( strat_test_set.head(5) )    
    
    # Let's make a copy so we have a set to play around with, but can easily
    # revert to the original
    
    housing = strat_train_set.copy()
    
    # view by geography. The resulting scatter-plot will look a lot like California
    # although no other patterns will really be apparent.
    # housing.plot( kind="scatter", x="longitude", y="latitude", grid=True )
    # plt.show()
    
    # visualize again with alpha=0.2 so we can see the high-density areas
    # housing.plot( kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2 )
    # plt.show()    
    
    # now use a color map to represent the prices, and circle size to represent the
    # district population, so we can see more patterns visually.
    # here we can see that prices tend to be higher near the coast and in high-density areas
    # housing.plot( kind="scatter", x="longitude", y="latitude", grid=True, 
    #                s=(housing["population"] / 100), label="population", c="median_house_value", cmap="jet", colorbar=True, legend=True, sharex=False, figsize=(10,7))
    # plt.show()
    
    
    # looking for correlations.
    # we can use Pearson's R (aka the "Standard correlation coefficient") to see how much one
    # attribute correlates to another. 
    # https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    # With SKLearn we can use the corr() method to generate all of the pairwise mappings and the
    # associated r value
    
    print( "Correlations:\n")
    
    corr_matrix = housing.corr()
    print( corr_matrix["median_house_value"].sort_values( ascending=False) )
    
    # We can also get a visual look at how features correlate to each other, by using the
    # scatter_matrix() function.
    
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age" ]
    # scatter_matrix(housing[attributes], figsize=(12,8))
    # plt.show()
    
    
    # let's zoom in on this specific attribute pair as it looks interesting
    # housing.plot( kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=TRUE )
    # plt.show()
    
    
    # Let's try making some new synthetic attributes using the existing attributes we have
    housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["people_per_house"] = housing["population"] / housing["households"]
    
    corr_matrix = housing.corr()
    print( corr_matrix["median_house_value"].sort_values(ascending=False) )
    
    housing = strat_train_set.drop( "median_house_value", axis=1 )
    housing_labels = strat_train_set["median_house_value"].copy()
    
    # Dealing with missing values. Our options include:
    # 1. get rid of the rows with missing values for feature X
    # 2. get rid of feature X completely
    # 3. Impute a value and fill in for the missing value. 
    #    Imputed values could be something like zero, the median of the existing 
    #    values, the mean of the existing values, etc.
    
    # One way to implement option 3 is as follows:
    # median = housing["total_bedrooms"].median()
    # housing["total_bedrooms"].fillna(median, inplace=True);
    
    
    # But we'll use SKLearn's SimpleImputer class since it makes life easier for us
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.select_dtypes(include=[np.number])
    imputer.fit( housing_num )
    
    print( "\n", imputer.statistics_, "\n")
    
    print( housing_num.median().values)
    
    X = imputer.transform(housing_num)
    
    housing_tr = pd.DataFrame( X, columns=housing_num.columns, index=housing_num.index  )
    
    
    print( "Info: (housing_tr) \n" )
    
    housing_tr.info()
        
    print( "Describe: (housing_tr) \n" )
    
    print( housing_tr.describe() )    
    
    housing_cat = housing[["ocean_proximity"]]
    print( "\n", housing_cat.head(8) )
    
    
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform( housing_cat )
    
    print( "\n", housing_cat_encoded[:8], "\n" )
    
    print( ordinal_encoder.categories_, "\n" )
    
    
    cat_encoder = OneHotEncoder(sparse=False)
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    
    
    print( housing_cat_1hot, "\n" )
    
    
    print( cat_encoder.categories_ )
    
    # Scaling - normalization and standardization
    
    # The MinMaxScaler will shift and rescale values so they fall into a range between
    # 0 and 1. This is done by subtracting the min value and dividing by the difference between the 
    # min and the max. 
    
    min_max_scaler = MinMaxScaler()
    housing_num_minMaxScaled = min_max_scaler.fit_transform( housing_num )
    
    print( "\n\n", type( housing_num_minMaxScaled ), "\n" )
    
    housing_num_minMaxScaled_df = housing_tr = pd.DataFrame( housing_num_minMaxScaled, columns=housing_tr.columns)
        
    print( "Info: (housing_num_minMaxScaled) \n" )
    
    housing_num_minMaxScaled_df.info()
        
    print( "Describe: (housing_num_minMaxScaled) \n" )
    
    print( housing_num_minMaxScaled_df.describe() )     
    
    print( "Head: (housing_num_minMaxScaled) \n" )
    print( housing_num_minMaxScaled_df.head() )
    
    
    # or, for a different range (say, -1 to 1) you can pass the feature_range argument to the 
    # MinMaxScaler ctor
    min_max_scaler = MinMaxScaler(feature_range=(-1,1))
    
    housing_num_minMaxScaled = min_max_scaler.fit_transform( housing_num )
    
    housing_num_minMaxScaled_df = pd.DataFrame( housing_num_minMaxScaled, columns=housing_tr.columns)
    
    print( "Head: (housing_num_minMaxScaled) \n" )
    print( housing_num_minMaxScaled_df.head() )
    
    # Another approach is to use the StandardScaler to do standardization.
    # This approach first subtracts the mean value (so standardized values have a mean of zero)
    # and then divides by the standard deviation (so standardized values have a stddev of 1).
    # Unlike min-max scaling, this does not restrict values to a specific range. 
    # Standardization is much less affected by outliers. 
    
    std_scaler = StandardScaler()
    
    housing_num_stdScaled = std_scaler.fit_transform(housing_num)
    
    housing_num_stdScaled_df = pd.DataFrame( housing_num_stdScaled, columns = housing_tr.columns )
    
    print( "Head: (housing_num_stdScaled) \n" )
    print( housing_num_stdScaled_df.head() )
    
    # Note: if a value is heavily skewed with a heavy tail, it's best to transform it into something
    # more symmetrical *before* scaling. 
    # A common way to do this for positive features with a heavy tail to the right is to replace
    # the value with its square root (or raise it to a power between 0 and 1). If the feature has
    # a *really* heavy and long tail, something close to a power-law distribution for example, you can
    # replace the value with its logarithm. 
    
    # let's do the log thing for the population of our dataset
    # For transformations like this one that don't require any training,
    # you can just write a function that takes a Numpy array as input and returns
    # the transformed Numpy array. Here we'll use the SKLearn FunctionTransformer class
    # to implement a simple LogTransformer that uses the np.log and np.exp functions
    # internally.
    
    log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
    
    log_pop = log_transformer.transform(housing[["population"]])
    
    
    # create a histogram here, showing the distribution of the original values
    # and the distribution of the new log'ified ones
    
    # plt.hist(housing[["population"]], bins=range(10000) ) # 
    # plt.show()
    
    # plt.hist( log_pop, bins=range(10) ) # 
    # plt.show()
    
    
    # for a custom transformer that actually has to learn using fit(), we need to write
    # a custom class. SKLearn uses duck typing, so we don't have to inherit from any specific
    # class. We just need to implement the three methods fit(), transform(), and
    # fit_transform()
    
    # We can, however, use TransformerMixin as a base class to get some default implementations to
    # help us out. Likewise, using BaseEstimator as a base class provides default support for
    # get_params() and set_params()
    
    # here's a custom transformer that is approximately the same as StandardScaler
    
    class StandardScalerClone( BaseEstimator, TransformerMixin ):
        def __init__(self, with_mean = True ):
            self.with_mean = with_mean
            
        def fit(self, X, y=None ):
            
            X = check_array(X) # checks that X is an array with finite float values
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            self.n_features_in_ = X.shape[1] # every estimator stores this in fit()
            return self # always returns self
            
            
        def transform(self, X):
            check_is_fitted(self) # looks for learned attributes (with trailing _)
            X = check_array(X)
            assert self.n_features_in_ == X.shape[1]
            if self.with_mean:
                X = X - self.mean_
            
            return X / self.scale_
        
    # end class StandardScalerClone
    
    # Pipelines
    # SKLearn provides a Pipeline class to help with making sure sequences of transformation steps
    # that need to occur in the right order are indeed performed so.
    
    # num_pipeline = Pipeline( [ ("impute", SimpleImputer(strategy="median")),
    #                           ("standardize", StandardScaler())] )
    

    num_pipeline = make_pipeline( SimpleImputer(strategy="median"), StandardScaler())
    
    housing_num_prepared = num_pipeline.fit_transform(housing_num)
    print( "\n", housing_num_prepared[:2].round(2))
    
    
    # again, to get back to a DataFrame, we can use the get_feature_names_out() method
    # df_housing_num_prepared = pd.DataFrame( housing_num_prepared, 
    #                                        columns=num_pipeline.get_feature_names_out(), 
    #                                        index=housing_num.index )
    
    
    # if we want to handle numerical and categorical features in a single transform step, we
    # can use ColumnTransformer and pass it two pipelines, one for numerical features and one for
    # categorical features.
    
    num_attribs = [ "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income" ]
    cat_attribs = [ "ocean_proximity" ]
    
    cat_pipeline = make_pipeline( SimpleImputer(strategy="most_frequent" ), OneHotEncoder(handle_unknown="ignore" ))
    
    preprocessing = ColumnTransformer([ ( "num", num_pipeline, num_attribs ), 
                                        ( "cat", cat_pipeline, cat_attribs ) ])
    
    
    housing_prepared = preprocessing.fit_transform( housing )
        
    df_housing_prepared = pd.DataFrame( housing_prepared, columns=preprocessing.get_feature_names_out(), index=housing.index)    
        
    # All together now, the condensed code to create a pipeline that does the following:
    # - missing values are replaced by the median value
    # - categorical features are one-hot encoded
    # - some ratio features are computed and added: bedrooms_ratio, rooms_per_house, people_per_house
    # - some cluster similarity features are added. Probably more useful that the raw longitude and latitude
    # - features with a long-tail are replaced by their logarithm
    # - all numerical features are standardized
    
    class ClusterSimilarity(BaseEstimator, TransformerMixin):
        def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
            self.n_clusters = n_clusters
            self.gamma = gamma
            self.random_state = random_state
        
        def fit(self, X, y=None, sample_weight=None):
            self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state, n_init=10)
            self.kmeans_.fit(X, sample_weight=sample_weight)
            return self # always return self!
        
        def transform(self, X):
            return rbf_kernel( X, self.kmeans_.cluster_centers_, gamma=self.gamma)
        
        def get_feature_names_out(self, names=None):
            return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
        
        
        
    def column_ratio(X):
        return X[:, [0]] / X[:,[1]]
    
    def ratio_name( function_transformer, feature_names_in):
        return ["ratio"]
    
    def ratio_pipeline():
        return make_pipeline( 
                SimpleImputer(strategy="median"),
                FunctionTransformer(column_ratio, feature_names_out=ratio_name),
                StandardScaler()
            )
    
    log_pipeline = make_pipeline( SimpleImputer(strategy="median"),
                                  FunctionTransformer(np.log, feature_names_out="one-to-one"),
                                  StandardScaler())
    
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
    default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    
    preprocessing = ColumnTransformer([
            ( "bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ( "rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ( "people_per_house", ratio_pipeline(), ["population", "households"]),
            ( "log", log_pipeline, [ "total_bedrooms", "total_rooms", "population", "households", "median_income"]),
            ( "geo", cluster_simil, ["latitude", "longitude"]),
            ( "cat", cat_pipeline, make_column_selector(dtype_include=object))],
       remainder=default_num_pipeline)  
      

    housing_prepared2 = preprocessing.fit_transform(housing)
    
    print( "\n\n", housing_prepared2.shape )
    
    print( preprocessing.get_feature_names_out())
    
        
    print( "\n\ndone" )
   

def is_id_in_test_set( identifier, test_ratio ):
    return crc32(np.int64(identifier)) < (test_ratio * (2**32))

def split_data_with_id_hash( data, test_ratio, id_column ):
    ids = data[id_column]
    in_test_set = ids.apply( lambda id_: is_id_in_test_set(id_, test_ratio ))
    return data.loc[~in_test_set], data.loc[in_test_set]


# see above, this version has the problem that it generates different sets on every run
def  shuffle_and_split_data( data, test_ratio ):
    
    shuffled_indices = np.random.permutation( len(data ) )
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
        
def load_housing_data():
    
    tarball_path = Path( "../../datasets/housing.tgz" )
    if not tarball_path.is_file():
        Path("../../datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path   )
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="../../datasets")
            
    return pd.read_csv(Path("../../datasets/housing/housing.csv"))

if __name__ == "__main__":
    main()
    
