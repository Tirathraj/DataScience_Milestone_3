import pandas as pd
import numpy as np
import ift6758.data.tidy_data as td



#Milestone 3 Function
def get_engineered_df_by_game(original_df , bool_list = False, save_csv=False) -> pd.DataFrame:

    """
    1. Retrieve a dataframe from tidy_data.py (regular and playoff combined) with all play_by_play events from start year to end year (inclusive)
    2. Append different features to the retrieved dataframe according to milestone 2 feature engineering 1 guidelines
    3. return full dataframe with new features appended
    4. option to save dataframe as a .csv file if save_csv boolean set to True

    Returns whole dataframe with newly engineered features for start to end seasons inclusive.

    Features and their details:
    
        x , y : x-coodinate and y_coordinate.
        zone_code : O (offensive), D (defensive), N (neutral) -> helps determine which side home/away team is defending
        event_owner_team_id : team responsible for event
        
        is_goal (engineered) -> check if event_type =='goal'
        empty_net (engineered) -> not 100% previse -> determined if goalie_id is missing during play

        x_mean_home (engineered) -> (utility field) -> checks the mean offensive position of home team. If mean is positive, home team is defending on left and attacking on right. Useful to determine home_defending_side later

        home_defending_side (engineered) -> missing for 2016-2018 season. therefore engineered using "x_mean_home".
        More details in "compute_home_defending_side" function

        x_distance (engineered) -> (utility field) -> determines x_axis distance from net/goal (89,0).
        Note that this is DIFFERENT TO DISTANCE_FROM_NET.
        We use x_distance and y coordinate to calculate euclidean distance to net.
        More details in "compute_row_x_distance" function. Needs to handle a lot of edge cases

        distance_net (engineered) -> EUCLIDEAN DISTANCE FROM NET TO SHOT -> IMPORTANT AND REQUIRED FOR MODEL TRAINING -> calculated using (x_distance , y).

        angle_deg (engineered) -> angle from net to shot. Need to consider 4 quadrants : 2 in front of goal and 2 behind goal.
        Range for front shots : [0,90] degrees. Range for back shots : [180, 270] degrees
        More details in "compute_row_shot_angle" function

        angle_rad (engineered) -> angle_deg converted to radians

    """
    df = original_df.copy()

    df = compute_is_goal(df)

    df= compute_empty_net(df)

    df = compute_home_defending_side(df)

    df = compute_df_shot_distance(df)

    df = compute_df_shot_angle(df)

    #Save as csv
    if(save_csv):
        df.to_csv(f"./features.csv", index=False)

    
    #print(df.head(20))
    #print(df.tail(20))
    
    return df




def get_engineered_df_by_season(start_year : int, end_year :int =-1 , bool_list = False, save_csv=False) -> pd.DataFrame:

    """
    1. Retrieve a dataframe from tidy_data.py (regular and playoff combined) with all play_by_play events from start year to end year (inclusive)
    2. Append different features to the retrieved dataframe according to milestone 2 feature engineering 1 guidelines
    3. return full dataframe with new features appended
    4. option to save dataframe as a .csv file if save_csv boolean set to True

    Returns whole dataframe with newly engineered features for start to end seasons inclusive.

    Features and their details:
    
        x , y : x-coodinate and y_coordinate.
        zone_code : O (offensive), D (defensive), N (neutral) -> helps determine which side home/away team is defending
        event_owner_team_id : team responsible for event
        
        is_goal (engineered) -> check if event_type =='goal'
        empty_net (engineered) -> not 100% previse -> determined if goalie_id is missing during play

        x_mean_home (engineered) -> (utility field) -> checks the mean offensive position of home team. If mean is positive, home team is defending on left and attacking on right. Useful to determine home_defending_side later

        home_defending_side (engineered) -> missing for 2016-2018 season. therefore engineered using "x_mean_home".
        More details in "compute_home_defending_side" function

        x_distance (engineered) -> (utility field) -> determines x_axis distance from net/goal (89,0).
        Note that this is DIFFERENT TO DISTANCE_FROM_NET.
        We use x_distance and y coordinate to calculate euclidean distance to net.
        More details in "compute_row_x_distance" function. Needs to handle a lot of edge cases

        distance_net (engineered) -> EUCLIDEAN DISTANCE FROM NET TO SHOT -> IMPORTANT AND REQUIRED FOR MODEL TRAINING -> calculated using (x_distance , y).

        angle_deg (engineered) -> angle from net to shot. Need to consider 4 quadrants : 2 in front of goal and 2 behind goal.
        Range for front shots : [0,90] degrees. Range for back shots : [180, 270] degrees
        More details in "compute_row_shot_angle" function

        angle_rad (engineered) -> angle_deg converted to radians

    """

    original_df = td.get_df_by_season(start_year , end_year, bool_list=False, save = save_csv )

    df = original_df.copy()

    df = compute_is_goal(df)

    df= compute_empty_net(df)

    df = compute_home_defending_side(df)

    df = compute_df_shot_distance(df)

    df = compute_df_shot_angle(df)

    
    if(end_year==-1):
        end_year=start_year

    #Save as csv
    if(save_csv):
    
        if(end_year==start_year): df.to_csv(f"./{start_year}_features.csv", index=False)

        else: df.to_csv(f"./{start_year}_{end_year}_features.csv" , index=False)

    
    print(df.head(20))
    print(df.tail(20))
    
    return df


#set boolean 1 if goal, 0 otherwise
def compute_is_goal(df : pd.DataFrame) -> pd.DataFrame:
    filter = (df['event_type']=='goal')
    df['is_goal'] = filter.astype(int)

    return df
    
#set boolean 1 if goal_id isnull, 0 otherwise (decent accuracy but can do better)

def compute_empty_net(df : pd.DataFrame) -> pd.DataFrame :
    filter = df['goalie_id'].isnull()
    df['empty_net'] = filter.astype(int)
    return df


def compute_home_defending_side( df : pd.DataFrame ) -> pd.DataFrame :

    """

    Determine mean offensive position of home team
    If mean +ve, home team is defending right and attacking left. And vice-versa

    Returns : a dataframe with 1 additonal field called df['x_mean_home'].
    The sign of this field is used to modify the field df['home_defending_side'] where it is null (seasons 2016-2018).

    """

    #Column used to group df
    col_names = ['game_id' , 'period_number' , 'event_owner_team_id' , 'zone_code']
    
    df_group = df.copy()
    
    #Filter offensive actions only
    zone_filter = df_group['zone_code'] == 'O'
    df_group = df_group[zone_filter]

    #Select home offensive actions only
    home_filter = df_group['event_owner_team_id']==df_group['home_id']
    df_group = df_group[home_filter]

    #Find mean of groups
    df_group = df_group.groupby(col_names)
    mean_group = df_group['x'].mean()     #Multi-indexed series
    #median_group = df_group['x'].median()

    #Reset multi-index series back to df
    mean_df = mean_group.reset_index(name = 'x_mean_home')
    #print(mean_df)

    #Use a left join to merge every play-by-play event with home_def_side
    #Merge using id and period_number
    mean_df.drop(columns=['zone_code' , 'event_owner_team_id'],inplace = True)

    merge_cols = ['game_id' , 'period_number']

    new_df = df.merge ( mean_df , on = merge_cols, how="left" )

    #if a home_defending_field is Nan, use mean to fill it.
    #Else leave unchanged

    missing_rows_filter =  new_df['home_defending_side'].isnull()


    #Series
    new_df['home_defending_side'] = new_df['home_defending_side'].astype('string')
    
    filled_rows = np.where ( new_df.loc[ missing_rows_filter, 'x_mean_home'] > 0 , 'left' , 'right' )

    new_df.loc[missing_rows_filter,'home_defending_side'] = filled_rows

    return new_df


def compute_df_shot_distance( df : pd.DataFrame ) -> pd.DataFrame :

    """
        Returns a df with 2 new fields:
            df['x_distance'] -> x-axis distance from an event to the net.
            df['distance_net'] -> euclidean distance (hypotenus) from event to the net -> DESIRED FEATURE

    """

    dist_df = df.copy()
    
    #dist_df['distance_net'] = dist_df.apply(lambda row : compute_row_shot_distance(row) , axis=1 )
    
    dist_df['x_distance'] = dist_df.apply(lambda row : compute_row_x_distance(row) , axis=1 )
    dist_df['distance_net'] = np.sqrt ( dist_df['x_distance'] **2 + dist_df['y'] **2) 

    return dist_df


def compute_row_x_distance(row):

    """

    - Get x-axis distance from an event to the net.
    - Used to calculate distance from net and angle from net

    4 cases:

    def_side = left:
        Offensive zone/ +ve Neutral zone : x_dist = 89 - x
        Defensive/ -ve Neutral zone : x_dist = 89 + x

    def_side = right:
        Offensive zone / -ve Neutral zone : x_dist = 89 - abs(x)
        Defensive/+ve Neutral zone : x_dist = 89 + abs(x)

    """
    
    x = row['x']
    y = row['y']

    side = row['home_defending_side'] #defending
    team_id = row['event_owner_team_id']
    zone = row['zone_code']

    if(team_id != row['home_id']):
        if(side =='left'): side='right'
        else: side ='left'

    x_dist=0
    
    if side=='left':
        if ( (zone=='O') | ( (zone=='N') & (x > 0.0) ) ):
            x_dist = np.abs( 89 - x ) #x_coord is always +ve  #abs here caters for shot in offensive zone behind goals
            
        else: #(zone = D or zone = N & x <= 0)
            x_dist = 89 + np.abs(x)

    #defending Side = right
    else:
        if( (zone=='O') | ( (zone=='N') & (x < 0.0) ) ):
            x_dist = np.abs ( 89 - np.abs(x) ) #abs caters for shot in -ve offensive zone behind goals

           #zone = D or (zone = N & x >= 0) 
        else:
            x_dist = np.abs( 89 + np.abs(x) )
            
    #dist = np.sqrt ( x_dist ** 2 + y**2 )

    row['x_dist'] = x_dist

    #return dist
    return x_dist
    


def compute_df_shot_angle( df : pd.DataFrame ) -> pd.DataFrame :

    """
        Uses "compute_row_shot_angle" to calculate angle row by row 
        Adds 2 new fields: angle in degrees and angle in radians.
        Returns df with the new fields

    """

    angle_df = df.copy()

    angle_df['angle_deg'] = angle_df.apply(lambda row : compute_row_shot_angle(row) , axis=1 )
    angle_df['angle_rad'] = np.radians(angle_df['angle_deg'])

    return angle_df

    
    

def compute_row_shot_angle(row) -> float:

    """Use arctan (opposite/adj) = arctan (y/x_dist) to calculate angle between shot and goal.
    In front of goal: angle restricted only between 0 and 90 degrees
    Behind goal : angle ranges from 180 to 270 degrees : (270 - arctan (y/x_dist) )

    If x_coordinate > 89 and zone = offensive, OR x_coordinate <-89 and zone= offensive : shot is taken from behind the goal.  

    Returns angle in Degrees
    
    """

    x = row['x']
    y = row['y']

    x_dist = row['x_distance']

    theta_degrees = 90

    #Prevent division by 0, return 90 deg immediately
    if(x_dist == 0):
        return theta_degrees

    theta_degrees = np.degrees( np.arctan ( np.abs(y) / np.abs(x_dist) ) )

    #Shot takes place behind goal
    #if ( (x >89 or x < -89) & row['zone_code']=='O' ):
    if ((x > 89 or x < -89) and row['zone_code'] == 'O'):
        theta_degrees = 270 - theta_degrees

    return theta_degrees 
    

if __name__ == "__main__":

    print(f"Still processig feature_eng_1 script...wait for completion message (order of minutes)")

    #Run this to get all data from start year (inclusive) to end year (inclusive) in 1 df with engineered features
    
    #df_16_19 = get_engineered_df_by_season(2016, 2019, save_csv=True)
    #print(df_16_19.head(10))
    #print(df_16_19.tail(10))
    #print(df_16_19.columns)    

    #Season-by-season
    
    #df_16 = get_engineered_df_by_season(2016,save_csv=True)
    #df_17 = get_engineered_df_by_season(2017,save_csv=True)
    #df_18 = get_engineered_df_by_season(2018,save_csv=True)
    #df_19 = get_engineered_df_by_season(2019,save_csv=True)
    #df_20 = get_engineered_df_by_season(2020,save_csv=True)
    #df_21 = get_engineered_df_by_season(2021,save_csv=True)

    #Range of seasons
    #df_17_18 = get_engineered_df_by_season(2017,2018,save_csv=True)


    print(f"End of feature_eng_1 script")