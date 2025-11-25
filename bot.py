import argparse
from cat_env import make_env
from training import train_bot
from utility import play_q_table

def main():
    parser = argparse.ArgumentParser(description='Train and play Cat Chase bot')
    parser.add_argument('--cat', 
                       choices=['mittens', 'batmeow', 'paotsin', 'peekaboo', 'squiddyboi', 'trainer',"ryan","angry","shy","copy","dustine","stalker"],
                       default='batmeow',
                       help='Type of cat to train against (default: mittens)')
    parser.add_argument('--render', 
                       type=int,
                       default=100,
                       help='Render the environment every n episodes (default: -1, no rendering)')
    
    args = parser.parse_args()
    
    # Train the agent
    #print(f"\n\nIteration {count}")

    #print(f"\nTraining agent against {args.cat} cat...")
    q_table, time = train_bot(
        cat_name=args.cat,
        render=args.render
    )
        
    #print("\nTraining complete! Starting game with trained bot...")
    #print("Press Q to quit.")
        
    # Play using the trained Q-table
    env = make_env(cat_type=args.cat)
    caught,moves = play_q_table(env, q_table, max_steps=60, window_title='Cat Chase - Final Trained Bot',move_delay=0.25)

    if caught:
        print(f"The bot successfully caught the cat! ||| {moves} ||| {time}")
    else:
        print("The bot failed to catch the cat this time.")
    

if __name__ == "__main__":
    main()
