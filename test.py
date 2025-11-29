import argparse
from cat_env import make_env
from training import train_bot
from utility import play_q_table

def main():
    parser = argparse.ArgumentParser(description='Train and play Cat Chase bot')
    parser.add_argument('--cat', 
                       choices=['mittens', 'batmeow', 'paotsin', 'peekaboo', 'squiddyboi', 'trainer',"ryan","angry","shy","copy","dustine","stalker","diagonal","pathfinder","echo","chaos","hunter"],
                       default='batmeow',
                       help='Type of cat to train against (default: mittens)')
    parser.add_argument('--render', 
                       type=int,
                       default=100,
                       help='Render the environment every n episodes (default: -1, no rendering)')
    
    args = parser.parse_args()

    choices=['mittens', 'batmeow', 'paotsin', 'peekaboo', 'squiddyboi', 'trainer',"ryan","angry","shy","copy","dustine","stalker","diagonal","chaos","hunter"]
    #choices=['angry']
    
    #choices = reversed(choices)
    # Train the agent
    for cat in choices:
        with open('newangrytest.txt', 'w') as f:
            print(f"\nCURRENT CAT: {cat.upper()}\n",file=f)
        print(f"\nCURRENT CAT: {cat.upper()}\n")
        count = 0
        while count <1:
            count+=1
            #print(f"\n\nIteration {count}")

            #print(f"\nTraining agent against {args.cat} cat...")
            q_table, time = train_bot(
                cat_name=cat,
                render=-1
            )
            
            #print("\nTraining complete! Starting game with trained bot...")
            #print("Press Q to quit.")
            
            # Play using the trained Q-table
            env = make_env(cat_type=cat)
            caught,moves = play_q_table(env, q_table, max_steps=60, window_title='Cat Chase - Final Trained Bot',move_delay=0.00)

            with open('newangrytest.txt', 'a') as f:
                if caught:
                    print(f"The bot successfully caught the cat! ||| {moves} ||| {time}")
                    print(f"The bot successfully caught the cat! ||| {moves} ||| {time}",file=f)
                else:
                    print("The bot failed to catch the cat this time.")
                    print("The bot failed to catch the cat this time.",file=f)
    

if __name__ == "__main__":
    main()
