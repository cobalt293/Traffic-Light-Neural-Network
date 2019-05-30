import keyboard
import datetime

#https://www.youtube.com/watch?v=1EiC9bvVGnk

start_time = datetime.datetime.utcnow()

def get_current_timestep():
    return (datetime.datetime.utcnow()-start_time).seconds

def same_timestep():
    current_timestep = (datetime.datetime.utcnow()-start_time).seconds
    return current_timestep == state_store[0]

def write_to_file(state_store):
    with open('traffic_light_data_entry.csv', 'a+') as f:
        new_line=str(state_store[0])+","+\
                 str(state_store[1])+","+\
                 str(state_store[2])+","+\
                 str(state_store[3])+","+\
                 str(state_store[4])+"\n"
        f.write(new_line)

state_store = [ get_current_timestep(), 0, 0, 0, 0]

continue_loop = True
while continue_loop:
    #update state_store if new timestep
    if not same_timestep():
        print("State: ",state_store)
        write_to_file(state_store)
        state_store = [ get_current_timestep(), 0, 0, 0, 0]

    if keyboard.is_pressed('d'):
        state_store[1] = 1
    if keyboard.is_pressed('f'):
        state_store[2] = 1
    if keyboard.is_pressed('j'):
        state_store[3] = 1
    if keyboard.is_pressed('k'):
        state_store[4] = 1
    if keyboard.is_pressed('p'):
        continue_loop=False