import os
import torch
import glob
import json
from actions import actions
from dqnAgent import DQNAgent
from pynput import keyboard
from datetime import datetime
import time
def get_latest_model_path(models_dir="models"):
    model_files = glob.glob(os.path.join(models_dir, "model_*.pth"))
    if not model_files:
        return None
    model_files.sort()  # Lexicographical sort works for timestamps
    return model_files[-1]
class KeyboardController:
    def __init__(self):
        self.should_exit = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char == 'q':
                print("\nTraining interrupted by user.")     
                self.should_exit = True
                global errorCode
                errorCode = 1
        except AttributeError:
            pass  # Special key pressed
    def set_exit(self, should_exit):
        self.should_exit = should_exit
    def is_exit_requested(self):
        return self.should_exit
def endWait(key):
    global waiting
    try:
        if(key.char == 'c'):
            waiting = False
    except:
        pass
def train(printActions = False):
    global waiting
    global errorCode
    errorCode = 0
    waiting = True
    try:
        
        listener = keyboard.Listener(on_press=endWait)
        listener.start()
        print("waiting for key c to be pressed to start training")  
        while waiting:
            time.sleep(0.1)
        os.system(f'osascript -e \'display notification "Train running" with title "skBot"\'')
        print("train running")
        actor = actions(False)
        agent = DQNAgent(actor.actionSize, actor.stateSize)
        os.makedirs("models", exist_ok=True)

        # Load latest model if available
        latest_model = get_latest_model_path("models")
        if latest_model:
            agent.load(os.path.basename(latest_model))
            # Load epsilon
            meta_path = latest_model.replace("model_", "meta_").replace(".pth", ".json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    agent.epsilon = meta.get("epsilon", 1.0)
                print(f"Epsilon loaded: {agent.epsilon}")
        controller = KeyboardController()
        episodes = 10000
        batch_size = 32


        for ep in range(episodes):
            if controller.is_exit_requested():
                break

            if(actor.getState()==-1):# err (multiple players or kings found)
                time.sleep(0.5)
                if(actor.getState(redo=True, minConf=0.85) == -1):
                    controller.set_exit(True)
                    break
                
            
            
            state = actor.state
            print(f"Episode {ep + 1} starting. Epsilon: {agent.epsilon:.3f}")  # <-- Add this line
            total_reward = 0
            done = False
            while (not done):
                os.system(f'osascript -e \'display notification "New action cycle" with title "skBot"\'')
                if controller.is_exit_requested():
                    print("Training interrupted by user.")
                    done = True
                    continue
                #print("state: ", state)
                
                os.system(f'osascript -e \'display notification "preforming getActions()" with title "skBot"\'')
                print("preforming getActions()")
                possibleStates = actor.getActions() #get possible states
                
                if(printActions == True):
                    allActions = ""
                    for i in range(len(possibleStates)):
                        allActions += (str(i) + " " + str(possibleStates[i]) + " | ")
                    print("all actions: " + allActions) #print all possible actions
                
                os.system(f'osascript -e \'display notification "agent.act()" with title "skBot"\'')
                print("preforming agent.act(state, possibleStates)")
                action = agent.act(state, possibleStates) #get the dqn's decided action
                print(f"actions is state {action} ammo is {actor.currentAmmo}")

                os.system(f'osascript -e \'display notification "actor.step()" with title "skBot"\'')
                print("preforming actor.step(action)")
                returnCode, reward, done, nextState = actor.step(action) #do the action and get reward and if it is done  
                if(returnCode == -1):
                    errorCode = -1
                    controller.set_exit(True)
                    print("exiting")
                    done = True
                    continue
                os.system(f'osascript -e \'display notification "remember + replay" with title "skBot"\'')
                print("preforming agent.remember()")
                agent.remember(state, action, reward, nextState, done) #dqn remembers
                print("preforming agent.replay()")
                agent.replay(batch_size)
                state = nextState
                total_reward += reward
                print("end of action cycle")
                print("\n\n")
            print("Episode ended")
            print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

            '''if ep % 10 == 0:
                agent.update_target_model()
                # Save model and epsilon every 10 episodes
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join("models", f"model_{timestamp}.pth")
                torch.save(agent.model.state_dict(), model_path)
                with open(os.path.join("models", f"meta_{timestamp}.json"), "w") as f:
                    json.dump({"epsilon": agent.epsilon}, f)
                print(f"Model and epsilon saved to {model_path}")'''
    except:
        errorCode = -1
    os.system(f'osascript -e \'display notification "Training ended with code {errorCode}" with title "skBot"\'')
    print(f"Training ended with code {errorCode}")

if __name__ == "__main__":
    train()

    