from os import environ
import autogen
from pathlib import Path

llm_config = {"api_key": environ.get('OPENAI_API_KEY'), "model": "gpt-4-1106-preview", "request_timeout": 800,}
gpt35_config = {"api_key": environ.get('OPENAI_API_KEY'), "model": "gpt-3.5-turbo"}

planner_message = '''
Planner. You are a planner for a Dungeons and Dragon's Dungeon Master. You will be given JSON representing the current state of the game and it is your job is to decide the high level next step of the game formatted as JSON. 
For instance if the party is currently in combat and there are more NPCs to fight, you will output:

    {
        "State": "Combat",
        "Turn": "PlayerA"
    }

Or if it is the NPCs turn to fight and their name is "Ranged Combatant 1" you would output:

    {
        "State": "Combat",
        "Turn": "Ranged Combatant 1"
    }

If the party is exploring and you want them to find a some point of interest, let's say a tavern, you would output:

    {
        "State": "Exploration",
        "Found": "Tavern"
    }

Of if the party is ambushed by NPCS you would output something like:

    {
        "State": "Exploration",
        "Event": "Ambushed",
        "Description": "3 level four orcs jump out of the bushes, 15 spaces away"
    }

Only ever reply in JSON. You may add keys to the JSON as you see fit. 
Be sure to keep the game exciting and surprising. Drawing on motifs from the high fantasy genre. 
ONLY CONSIDER THE JSON FROM THE SUMMARIZER. DO NOT CHEAT AND LOOK AT ANYTHING FROM THE NARRATOR, CRITIC, or PLAYER!!
'''

initial_state = Path("./state.json").read_text()

json_summary_message = '''
Summarizer. You always speak after the Outcome_Decider. You summarize the current state of the Dungeons and Dragons game as JSON including all character stats and anything relevant for future dungeon masters to pick up if we leave the game.
Be extremely verbose in your representation so that we can pick up where we left off in the future. Maintain all details about the characters stats.
YOU ONLY EVER SPEAK IN JSON FORMAT. NO ENGLISH EVER. BE VERBOSE.
'''

narrator_message = '''
Narrator. You always speak after the Planner. Given JSON for the next state of the dungeons and dragons game from the Planner you give a human friendly narration. If it's the player's turn you prompt for their action. Never output JSON. Always output human friendly narration.
'''

critic_message = '''
Critic. You always speak after the Narrator. Double check Narrator for conformance to Dungeons and Dragon's 5th edition rules. Check wether the player or environment needs to make a dice roll.
'''

outcome_message = '''
Outcome_Decider. You always speak after the Player. Given the players actions and the scenario, decide the outcome of a turn. If a combat phase has ended include any XP earned or inventory items acquired and provide that to the Summarizer for JSON summary.
YOU SHOULD DESCRIBE AN ENTIRE TURN in DND. DO NOT DESCRIBE A PARTIAL TURN.
'''

player_message = '''
Player. You speak after the Critic. You tell the Outcome Decider your actions or choices.
'''

planner = autogen.AssistantAgent("Planner", 
        llm_config=llm_config, 
        system_message=planner_message)
summarizer = autogen.AssistantAgent("Summarizer",
        llm_config=llm_config,
        system_message=json_summary_message)
narrator = autogen.AssistantAgent("Narrator",
        llm_config=llm_config,
        system_message=narrator_message)
dnd_critic = autogen.AssistantAgent("Critic", 
        llm_config=llm_config,
        system_message=critic_message)

outcome = autogen.AssistantAgent("Outcome_Decider", 
        llm_config=llm_config,
        system_message=outcome_message)

player = autogen.UserProxyAgent("Player", 
        llm_config=llm_config,
        system_message=player_message)

groupchat = autogen.GroupChat(agents=[planner, narrator, dnd_critic,player, outcome, summarizer ], messages=[], max_round=7)

dungeon_master = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

summarizer.initiate_chat(dungeon_master, message=initial_state)

state = groupchat.messages.pop()['content']
state = state.replace("```json", "")
state = state.replace("```", "")
f = open('state.json', 'w')
f.write(state)
