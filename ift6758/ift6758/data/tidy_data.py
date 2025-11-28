import pandas as pd
import json
from pathlib import Path
import numpy as np
import os

#from path_variables import DATA_DIR


def load_by_id(year : int , game_type : int , game_number : int ) -> dict :
    """

    Load json file from disk. Make sure data is in same directory as this file.
-----------------------------------------------------------------------------------------------------------------------------------------------
    Args:
        year (int): season
        game_type (int): 2 for regular game, or 3 playoff game
        game_number (int) : function handles conversion to 4 decimal places : e.g 1 -> 001 etc.


    Returns:
        dict: a nested dict containing meta/root data and play-by-play events about the particular game.
------------------------------------------------------------------------------------------------------------------------------------------------

    Reference: Mastering Object-Oriented Python (O’Reilly)
    Reference Link: https://www.oreilly.com/library/view/mastering-object-oriented-python/9781789531367/c34be237-5ccd-4775-a0b0-ec1f7652f7bc.xhtml?
    
    """
    
    #BASE_DIR=Path("./")
    #BASE_DIR=Path("./ift6758/data")

    #BASE_DIR=("C:\\Users\\rambu\\Python Scripts\\Data Science IFT 6758\\Project 2025\\project-template\\ift6758\\data")
    
    #print(BASE_DIR)

    #BASE_DIR = Path(__file__).resolve().parent #fix path issues
    
    BASE_DIR = DATA_DIR
    

    game_id=f"{year}{game_type:02d}{game_number:04d}"

    if (game_type==2):
        file_path = BASE_DIR / Path(f"{year}") / Path("Regular") / Path(f"{game_id}.json")
        
    else: file_path = BASE_DIR / Path(f"{year}") / Path("Playoffs") / Path(f"{game_id}.json")

    if not file_path.exists():
        print(f"Invalid Id: {game_id}, {file_path}")
        return

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data #Data is of type dict (nested)


def get_root_data_by_game( data : dict) -> dict :

    """
    Get root metadata for a game.
    Flattens the nexted json dict structure to include metadata with every play-by-play event.
-------------------------------------------------------------------------------------------------
    Args:
        data (dict) : Takes a json dictionary
        This dictionary is obtained by calling load_by_id() function above
-------------------------------------------------------------------------------------------------
    Returns:
        4 different dictionaries with different metadata about game, venue, home and away teams.
        game_info dict contains the most important information about every game

    """

    game_info={

        'game_id' : data['id'],
        'season' : data['season'],
        'game_type' : data['gameType'],
        'game_outcome': data['gameOutcome']['lastPeriodType']
    }

    venue_info={
        
        'venue' :  data['venue']['default'],
        'venue_location' : data['venueLocation']['default']
        #'start_time' : data['startTimeUTC']
    }

    #Set home team dict to a variable to limit nesting
    #Build home team dict

    home=data['homeTeam']
    home_team = {

        'home_id' : home['id'],
        'home_name': home['commonName']['default'],
        'home_location': home['placeName']['default'],
        'home_abrev': home['abbrev'],
        'home_score': home['score']
    }

    #Set away team dict to a variable to limit nesting
    #Build away team dict

    away=data['awayTeam']
    away_team = {

        'away_id' : away['id'],
        'away_name': away['commonName']['default'],
        'away_location': away['placeName']['default'],
        'away_abrev': away['abbrev'],
        'away_score': away['score']
    }
        

    return game_info , venue_info, home_team, away_team


#Return a Dataframe for a single game
#Loads json, retrieves root metadata and combines with play-by-by events
def get_df_by_game(year : int , game_type : int , game_number : int, bool_list = False ) -> pd.DataFrame:

    """
    One of the most important functions : Will be extensively called for every game 
    Modular design:

    1. For one game: we first it's respective load data and get a json dict.
    
    2. We then call get_play_by_play_by_game(data) to combine game metadata and play-by-play (Flattening).
    
    3. We retrieve a list of play-by-play events for that game and either return it, or create a df from that and return it.

    --------------------------------------------------------------------------------

    Args:
        year (int) : season
        game_type (int): 2 for regular games, 3 for playoffs
        game_number (int)
        bool_list (boolean): if true, function simply returns list of events instead of dataframe

    ---------------------------------------------------------------------------------

    Returns:
        DataFrame or List of Events for 1 game only    

    """
    
    data=load_by_id(year,game_type,game_number)
    #print(data.keys())

    list_of_plays = get_play_by_play_by_game (data)

    df = pd.DataFrame( list_of_plays )

    if(bool_list):
        return list_of_plays

    return df

def get_df_by_game_all_plays(year : int , game_type : int , game_number : int, bool_list = False ) -> pd.DataFrame:

    """
    One of the most important functions : Will be extensively called for every game 
    Modular design:

    1. For one game: we first it's respective load data and get a json dict.
    
    2. We then call get_play_by_play_by_game_all_plays(data) to combine game metadata and play-by-play (Flattening).
    
    3. We retrieve a list of play-by-play events for that game and either return it, or create a df from that and return it.

    --------------------------------------------------------------------------------

    Args:
        year (int) : season
        game_type (int): 2 for regular games, 3 for playoffs
        game_number (int)
        bool_list (boolean): if true, function simply returns list of events instead of dataframe

    ---------------------------------------------------------------------------------

    Returns:
        DataFrame or List of Events for 1 game only    

    """
    
    data=load_by_id(year,game_type,game_number)
    #print(data.keys())

    list_of_plays = get_play_by_play_by_game_all_plays(data)

    df = pd.DataFrame( list_of_plays )

    if(bool_list):
        return list_of_plays

    return df

def add_homeDefSide(df : pd.DataFrame) -> pd.DataFrame:
    '''
    Function to populate the "home_defending_side" column since it was only introduced after 2020. 
    We use the first "shot-on-goal" event of each game to determine which side is defended by the home team. 
    This however does not work for shots that were made behind the center line.
    Args:
        df: dataframe to populate.
    Returns:
        dataframe with a populated "home_defending_side" column indicating the side of the rink the home team is defending.
    '''
    game_ids = df['game_id'].unique()
    for game_id in game_ids:
        df_game = df[df['game_id'] == game_id]
        first_shot = df_game[df_game['event_type'] == 'shot-on-goal'].iloc[0]
        if (first_shot['x'] > 0 and first_shot['event_owner_team_id'] == first_shot['home_id']) or (first_shot['x'] < 0 and first_shot['event_owner_team_id'] == first_shot['away_id']):
            df.loc[(df['game_id'] == game_id) & (df['period_number'] % 2 == 1), 'home_defending_side'] = 'left'
            df.loc[(df['game_id'] == game_id) & (df['period_number'] == 2), 'home_defending_side'] = 'right'
        else:
            df.loc[(df['game_id'] == game_id) & (df['period_number'] % 2 == 1), 'home_defending_side'] = 'right'
            df.loc[(df['game_id'] == game_id) & (df['period_number'] == 2), 'home_defending_side'] = 'left'
    return df.copy()



def get_player_details(data : dict) -> dict:
    """
    Builds a dict of player & their names etc, and returns it

    Args: json dictionary

    Returns: dictionary mapping player_id to various attributes

    """

    players={}
    
    roster = data['rosterSpots']

    for pl in roster:

        p_id=pl['playerId']
        
        info={}
        info['team_id'] = pl['teamId']
        info['first_name'] = pl['firstName']['default']
        info['last_name'] = pl['lastName']['default']
        info['position_code'] = pl['positionCode']

        players[p_id] = info

    return players


def get_event_owner_team_name(owner_id : int, home: dict , away: dict) -> str:
    """
    Utility function to retrieve team name of offensive players during play0by-play

    Args: Uses 2 of the dictionaries returned by get_root_data_by_game() function.

    Returns: Associated team name as string
        
    """
    
    home_id=home['home_id']
    home_name=home['home_name']

    if (home_id==owner_id):
        return home_name

    return away['away_name']
    

#Only shots and goals for now, as indicated in Milestone 1 document

def get_play_by_play_by_game(data : dict ) -> list[dict]:

    """

    MOST IMPORTANT FUNCTION FOR 1 GAME
    
    1. Retrieve game metadata FOR 1 GAME.
    2. Loop through & Filter play events and INCLUDE ONLY SHOTS-ON-GOAL AND GOAL (Milestone 1).
    3. Flatten structure so that metadata propagates to every play event.
    4. handles missing values using np.nan.
    5. Do some python dictionary unpacking to alter order of columns in dataframe.

    Called in: 

    Args:
        json dict

    Returns:
        A list of dicts of play-by play events merged with flattened root data -> Flattened structure

    """

    #Get metadata
    game, venue, home, away= get_root_data_by_game(data)

    event_list=['shot-on-goal','goal', 'missed-shot']

    #Retrieve all players details from 'rosterSpots' dict
    player_details = get_player_details(data)
    
    list_of_plays=[]

    plays=data['plays']

    i=0
    
    for play in plays:

        #If event is not a shot-on-goal or goal, skip
        event_type = play['typeDescKey']
        
        if event_type not in event_list:
            continue

        details=play['details']

        event_owner_id=details.get('eventOwnerTeamId', np.nan)

        event_owner_team_name = np.nan

        if not pd.isna(event_owner_id):
            event_owner_team_name = get_event_owner_team_name(event_owner_id,home,away)
        
        goalie_id = details.get('goalieInNetId', np.nan)
        player_id = details.get('shootingPlayerId', np.nan)

        goalie_name = np.nan
        player_name= np.nan

        if not pd.isna(player_id):
            player_name=f"{player_details[player_id]['first_name']} {player_details[player_id]['last_name']}"

        if not pd.isna(goalie_id):
            goalie_name=f"{player_details[goalie_id]['first_name']} {player_details[goalie_id]['last_name']} "
            
        #Player Key name changes in case of a goal 
        
        if (event_type==event_list[1]):
            player_id = details.get('scoringPlayerId', np.nan)
            player_name=f"{player_details[player_id]['first_name']} {player_details[player_id]['last_name']}"
            
        #  **dict: Unpacks dicts in logical order
        
        play_info = {

        **game,
        'period_number': play.get('periodDescriptor').get('number'),
        'period_type': play.get('periodDescriptor').get('periodType'),
        'time_in_period': play.get('timeInPeriod', np.nan),
        'strength': details.get('strength', np.nan),
        'event_type': event_type,
        'shot_type': details.get('shotType', np.nan),
        'x': details.get('xCoord'),
        'y': details.get('yCoord'),
        'player_id': player_id,
        'player_name': player_name,
        'goalie_id': goalie_id,
        'goalie_name': goalie_name,
        'zone_code': details.get('zoneCode'),
        'event_owner_team_id': event_owner_id,
        'event_owner_team_name': event_owner_team_name,
        'home_sog': details.get('homeSOG', np.nan),
        'away_sog':details.get('awaySOG', np.nan),
        'home_defending_side': play.get('homeTeamDefendingSide',np.nan),

        **home,
        **away,
        **venue
        
        } #end of play_info dict

        list_of_plays.append(play_info) #list of dicts
    
    return list_of_plays

def get_play_by_play_by_game_all_plays(data : dict ) -> list[dict]:

    """

    MOST IMPORTANT FUNCTION FOR 1 GAME
    
    1. Retrieve game metadata FOR 1 GAME.
    2. Include all plays.
    3. Flatten structure so that metadata propagates to every play event.
    4. handles missing values using np.nan.
    5. Do some python dictionary unpacking to alter order of columns in dataframe.

    Called in: 

    Args:
        json dict

    Returns:
        A list of dicts of play-by play events merged with flattened root data -> Flattened structure

    """

    #Get metadata
    game, venue, home, away= get_root_data_by_game(data)

    #Retrieve all players details from 'rosterSpots' dict
    player_details = get_player_details(data)

    event_list=['shot-on-goal','goal', 'missed-shot']
    
    list_of_plays=[]

    plays=data['plays']

    i=0
    
    for play in plays:

        event_type = play['typeDescKey']

        try:
            details=play['details']
        except:
            # event has no details
            details = {}
            
        event_owner_id=details.get('eventOwnerTeamId', np.nan)

        event_owner_team_name = np.nan

        if not pd.isna(event_owner_id):
            event_owner_team_name = get_event_owner_team_name(event_owner_id,home,away)
    
        goalie_id = details.get('goalieInNetId', np.nan)
        player_id = details.get('shootingPlayerId', np.nan)

        goalie_name = np.nan
        player_name= np.nan

        if not pd.isna(player_id):
            player_name=f"{player_details[player_id]['first_name']} {player_details[player_id]['last_name']}"

        if not pd.isna(goalie_id):
            goalie_name=f"{player_details[goalie_id]['first_name']} {player_details[goalie_id]['last_name']} "
            
        #Player Key name changes in case of a goal 
        
        if (event_type=="goal"):
            player_id = details.get('scoringPlayerId', np.nan)
            player_name=f"{player_details[player_id]['first_name']} {player_details[player_id]['last_name']}"
        
        duration = np.nan
        penalty_type = ''
        if (event_type == 'penalty'):
            duration = details.get('duration', np.nan)
            penalty_type = details.get('typeCode', '')

            
        #  **dict: Unpacks dicts in logical order
        
        play_info = {

        **game,
        'period_number': play.get('periodDescriptor').get('number'),
        'period_type': play.get('periodDescriptor').get('periodType'),
        'time_in_period': play.get('timeInPeriod', np.nan),
        'strength': details.get('strength', np.nan),
        'event_type': event_type,
        'shot_type': details.get('shotType', np.nan),
        'x': details.get('xCoord'),
        'y': details.get('yCoord'),
        'penalty_duration' : duration * 60,
        'penalty_type' : penalty_type,
        'player_id': player_id,
        'player_name': player_name,
        'goalie_id': goalie_id,
        'goalie_name': goalie_name,
        'zone_code': details.get('zoneCode'),
        'event_owner_team_id': event_owner_id,
        'event_owner_team_name': event_owner_team_name,
        'home_sog': details.get('homeSOG', np.nan),
        'away_sog':details.get('awaySOG', np.nan),
        'home_defending_side': play.get('homeTeamDefendingSide',np.nan),

        **home,
        **away,
        **venue
        
        } #end of play_info dict

        list_of_plays.append(play_info) #list of dicts
    
    return list_of_plays



def get_file_names_per_season( year : int, reg_or_playoffs = 'both') -> list[Path] : 
    """
        Returns list of path of all json files in a season directory (Regular, THEN Playoff -> order matters here)

        Args:
            season (int)

        Returns:
            list of file paths in a 'year' folder.

    """

    BASE_DIR = DATA_DIR
    #BASE_DIR = Path(__file__).resolve().parent #fix path issues

    #BASE_DIR= Path("./")
    #BASE_DIR=Path("./ift6758/data")

    #BASE_DIR=("C:\Users\rambu\Python Scripts\Data Science IFT 6758\Project 2025\project-template\ift6758\data")
    #BASE_DIR=("C:\\Users\\rambu\\Python Scripts\\Data Science IFT 6758\\Project 2025\\project-template\\ift6758\\data")

    print(f"File dir: {BASE_DIR}")
    print("In files function")
    
    season_path_reg = BASE_DIR/Path(f"{year}")/Path(f"Regular")
    season_path_playoffs = BASE_DIR/Path(f"{year}")/Path(f"Playoffs")

    print(f"{season_path_reg}")
    print(f"{season_path_playoffs}")

    file_list_reg=season_path_reg.glob("*.json")
    file_list_playoff=season_path_playoffs.glob("*.json")

    #print(file_list_reg)

    all_files = []

    if (reg_or_playoffs=='reg') or (reg_or_playoffs=='both'):
        for file in file_list_reg:
            all_files.append(file)
        
    if (reg_or_playoffs=='playoff') or (reg_or_playoffs=='both'):
        for file in file_list_playoff:
            all_files.append(file)

    return all_files

    """
    This code snippet does what the ABOVE function does, however Playoff data comes first which is undesirable.
    Hence commented out.
    
    season_path=BASE_DIR/Path(f"{year}")

    # Recursively retrieve json files from every folder present for a season (Reg & Playoffs Folders)
    file_list=season_path.glob("**/*.json") 

    return file_list

   """ 



    

def get_df_by_season(start_year : int, end_year :int =-1 , bool_list=False, save=False) -> pd.DataFrame:
    """
    One of the most important functions.
    1. Retrieve file paths for a season/range of season
    2. Loop through all files and call get_df_by_game to retrieve data for every game.
    3. Go through all games AND STORE RESULTS IN A BIG LIST OF DICTS

    4. Return that list, OR RETURN A DATAFRAME OF THAT LIST, depending on parameter bool_list.
    
    5. Possibility to save all the play-by-play events in a CSV by specifying (save = True ) in the function argument

    **N.B Cannot save or return list at the same time: One or the other
-------------------------------------------------------------------------------------------------------------------

    Args:
        start_year (int) : MANDATORY ARGUMENT, (inclusive)
        end_year (int) : Optional
        bool_list (boolean) : return a Dataframe of play-by-play for whole season, or a list of play-by-play
        save (boolean) : save as csv file AND then return dataframe

    Returns:
        Dataframe OR list of events depending on parameter     

    """

    if(end_year==-1):
        end_year=start_year
        
    #List that will contain rows of play-by-play for the whole season
    all_plays = []

    for year in range(start_year , end_year + 1 , 1):
        print(f"Loading DF for {year}")
        
        #List of Windows Path

        all_files = get_file_names_per_season(year)
        print(f"Getting files done")

        for file in all_files:
            
            year=file.parent.parent.name
            g_type=file.parent.name
            g_id=file.stem
    
            g_type_code = 2
    
            if(g_type=="Playoffs"): g_type_code = 3
    
            g_num = int(g_id[-4:])

            #Keep track of progress
            if(g_num%100==0): print(year, g_type,g_id, g_num)
            
            list_of_plays = get_df_by_game(year, g_type_code, g_num, bool_list=True) #Very important function call
    
            all_plays.extend(list_of_plays)
        #end of file iteration : every file represents one game of the season
        
    #end of year or season iteration

    #----
    
    if(bool_list): 
        return all_plays   #Return events as a list
    
    df = pd.DataFrame(all_plays)

    if not save:
        return df

    #Save to CSV
    
    #if(end_year==start_year): df.to_csv(f"./{start_year}.csv")

    #else: df.to_csv(f"./{start_year}_{end_year}.csv" , index=False)

    if(end_year==start_year): df.to_csv(f"./{start_year}.csv", index=False)

    else: df.to_csv(f"./{start_year}_{end_year}.csv" , index=False)
    
    return df

def get_df_by_season_all_plays(start_year : int, end_year :int =-1 , bool_list=False, save=False, reg_or_playoffs = 'both') -> pd.DataFrame:
    """
    One of the most important functions.
    1. Retrieve file paths for a season/range of season
    2. Loop through all files and call get_df_by_game to retrieve data for every game.
    3. Go through all games AND STORE RESULTS IN A BIG LIST OF DICTS

    4. Return that list, OR RETURN A DATAFRAME OF THAT LIST, depending on parameter bool_list.
    
    5. Possibility to save all the play-by-play events in a CSV by specifying (save = True ) in the function argument

    **N.B Cannot save or return list at the same time: One or the other
-------------------------------------------------------------------------------------------------------------------

    Args:
        start_year (int) : MANDATORY ARGUMENT, (inclusive)
        end_year (int) : Optional
        bool_list (boolean) : return a Dataframe of play-by-play for whole season, or a list of play-by-play
        save (boolean) : save as csv file AND then return dataframe

    Returns:
        Dataframe OR list of events depending on parameter     

    """

    if(end_year==-1):
        end_year=start_year
        
    #List that will contain rows of play-by-play for the whole season
    all_plays = []

    for year in range(start_year , end_year + 1 , 1):
        print(f"Loading DF for {year}")
        
        #List of Windows Path

        all_files = get_file_names_per_season(year, reg_or_playoffs=reg_or_playoffs)
        print(f"Getting files done")

        for file in all_files:
            
            year=file.parent.parent.name
            g_type=file.parent.name
            g_id=file.stem
    
            g_type_code = 2
    
            if(g_type=="playoffs"): g_type_code = 3
    
            g_num = int(g_id[-4:])

            #Keep track of progress
            if(g_num%100==0): print(year, g_type,g_id, g_num)
            
            list_of_plays = get_df_by_game_all_plays(year, g_type_code, g_num, bool_list=True) #Very important function call
    
            all_plays.extend(list_of_plays)
        #end of file iteration : every file represents one game of the season
        
    #end of year or season iteration

    #----
    
    if(bool_list): 
        return all_plays   #Return events as a list
    
    df = pd.DataFrame(all_plays)

    if not save:
        return df

    #Save to CSV
    
    #if(end_year==start_year): df.to_csv(f"./{start_year}.csv")

    #else: df.to_csv(f"./{start_year}_{end_year}.csv" , index=False)

    if(end_year==start_year): df.to_csv(f"./{start_year}.csv", index=False)

    else: df.to_csv(f"./{start_year}_{end_year}.csv" , index=False)
    
    return df

def get_shot_events_by_season(start_year : int, end_year :int =-1 , bool_list=False, save=False) -> pd.DataFrame:
    """
    Function to retrieve only shot events for a season/range of seasons.
    Calls get_df_by_season and then filters the dataframe to only include shot events.
    Shot events are defined as events where typeDescKey contains "shot-on-goal", "goal", or "missed-shot".
    This does not include: blocked-shots, faceoffs, hits, giveaways, takeaways, penalties, etc.
    
    Args:
        start_year (int) : MANDATORY ARGUMENT, (inclusive)
    
        end_year (int) : Optional
        bool_list (boolean) : return a Dataframe of play-by-play for whole season, or a list of play-by-play
        save (boolean) : save as csv file AND then return dataframe

    Returns:
        Dataframe OR list of shot events depending on parameter"""
    if os.path.exists(f"./{start_year}_shots.csv"):
        return pd.read_csv(f"./{start_year}_shots.csv")
    elif os.path.exists(f"./{start_year}_{end_year}_shots.csv"):
        return pd.read_csv(f"./{start_year}_{end_year}_shots.csv")
    else:
        df = get_df_by_season(start_year, end_year, bool_list=False, save=False)

        df_shots = df[df['event_type'].str.contains('shot-on-goal|goal', case=False, na=False)].copy()
        df_shots = add_homeDefSide(df_shots)

        if bool_list:
            return df_shots.to_dict('records')

        if save:
            if(end_year==start_year or end_year == -1): df_shots.to_csv(f"./{start_year}_shots.csv", index=False)

            else: df_shots.to_csv(f"./{start_year}_{end_year}_shots.csv" , index=False)

        return df_shots
    
def get_team_ids_list() -> list[str]:
    """
    Function to retrieve all team ids in the data.
    Returns:
        list of team names.
    """
    team_ids = set()
    for year in range(2016, 2024):
        df = get_df_by_season(year, bool_list=False, save=False)
        team_ids.update(df['home_id'].unique())
        team_ids.update(df['away_id'].unique())
    return sorted(list(team_ids))

def get_team_names_list() -> list[str]:
    """
    Function to retrieve all team names in the data.
    Returns:
        list of team names.
    """
    team_names = set()
    for year in range(2016, 2024):

        df = get_df_by_season(year, bool_list=False, save=False)

        #print(df)
        
        team_names.update(df['home_name'].unique())
        team_names.update(df['away_name'].unique())
    return sorted(list(team_names))

if __name__=="__main__":


    """
    VERY IMPORTANT FUNCTIONS; Examples below:
    
        1. load_by_id -> load data from json.
        
        2,3. get_df_by_game -> Get df/ list of play-by-play events for a single game.
        
        4,5,6,7.  get_df_by_season -> use same function with different params to get different results.
                    e.g: single season, range of seasons, save to csv, get a list of events for whole seasons (instead of df)
                    Details and examples below

-------------------------------------------------------------------------------------------------------------------------------------------
        Additional Utility Functions:
        
        0. Make sure to load json and get raw data dict first -> used as param for most of these:

        1. get_player_details : takes json dict for a game, builds a dict of player & their names etc, and returns it

        2. get_root_data_by_game : takes json dict, retrieves general game metadata that is propagated to all play-by-play data
        
                returns 4 dicts: game_info (VERY IMPORTANT: id, reg/playoff, etc)
                                venue_info , home_info , away_info

        4. get_play_by_play_by_game : takes json dict, and returns all play_by_play events of type shot-in-goal/goal as a list
        
        
        
    """
    
    #1. Returns raw json data in dictionary form only as found in files
    
    data_dict=load_by_id(2022,2,1)
    #print(data_dict.keys())

    #2. Tidies data and returns a dataframe for 1 game only 
    
    df_1 = get_df_by_game(2022,2,1) #Regular Game (middle param = game_type = 2/3, end param: last 4 digits of game_id)
    print(df_1)

    #3. Very Important : Use same function but with an optional param (bool_list = True) to get a list of play-by-play events
    # Helpful and can be used directly for stuff.
    
    list_of_plays = get_df_by_game (2017,3,111, bool_list = True)
    #print(len(list_of_plays))

    
    #4. Get dataframe (both reg/playoffs combined) for a whole season
    df_season = get_df_by_season ( 2020, save = True )    
    print(df_season.columns, len(df_season))

    #5. Get dataframe for a range of season: start_year & end_year BOTH INCLUSIVE
    #df_16_18 = get_df_by_season ( 2016 , 2018 )

    #6. USE PARAMETER (save = True) TO SAVE TO CSV FILE -> Important
    
    df_16_17 = get_df_by_season ( 2016 , 2017 , save=True )
    print(f"N Rows: {len(df_16_17)}")

    #7. USE PARAMETER (bool_list=True) TO GET a list of play-by-play events instead of a dataframe
    
    #season_list_of_plays = get_df_by_season (2020 , 2024, bool_list = True)
    #print(len(season_list_of_plays))
    
    
    #8. Get all data and save it
    
    #df_all = get_df_by_season(2016,2024,save=True)
    

    #---------------------------------------------------------------------------------------------------------------------------------------------
       
        #Helpful Utility Functions
    
    #data_dict=load_by_id(2022,2,1) #Raw json file
    
    #game_info , venue_info, home_team, away_team = get_root_data_by_game(data_dict)
    #print(game_info)

    #player_details = get_player_details(data_dict)
    #print(len(player_details))

    #play_events = get_play_by_play_by_game (data_dict)
    #print(play_events[0])

    
    

    

    
    

    

    
