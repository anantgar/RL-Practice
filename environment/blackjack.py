from .environment import Environment
import numpy as np

class Blackjack(Environment):

    num_states = 10 * 2 * 10
    num_actions = 2

    def __init__(self):
        count = 0
        usable_ace = False
        while count < 12:
            card = self.draw_card()
            if card == 1:
                usable_ace = True
                count += 1 if count + 11 > 21 else 11
            else:
                count += card
        
        dealer_card = self.draw_card()
        self.state = (count, usable_ace, dealer_card)

    def draw_card(self):
        return np.random.choice(np.arange(1, 11), replace=True, p=[1/14] * 9 + [5/14])

    def step(self, action:int):
        if action == 0: # stick
            dealer_count = self.simulate_dealer()
            if dealer_count > 21 or dealer_count < self.state[0]:
                reward = 1
            elif dealer_count > self.state[0]:
                reward = -1
            else:
                reward = 0.5
            return None, reward, True

        elif action == 1: # hit
            count = self.state[0]
            usable_ace = self.state[1]
            card = self.draw_card()
            new_count = count + card
            if new_count > 21:
                if usable_ace:
                    new_count -= 10
                    usable_ace = False
                else:
                    return None, -1, True
            
            self.state = (new_count, usable_ace, self.state[2])
            return self.get_state(), 0, False
        else:
            raise ValueError(f"Invalid action: {action}")

    def simulate_dealer(self):
        count = self.state[2]
        usable_ace = count == 1
        while count < 17:
            card = self.draw_card()
            if card == 1:
                usable_ace = False if count + 11 > 21 else usable_ace
                count += 11 if usable_ace else 1
            else:
                new_count = count + card
                if new_count > 21:
                    if usable_ace:
                        new_count -= 10
                        usable_ace = False
                    else:
                        return -1
                count = new_count
        return count

    def set_state(self, state:int):
        self.state = self.deserialize_state(state)

    def get_state(self):
        return self.serialize_state(self.state)

    def get_random_state(self):
        return np.random.randint(0, self.num_states)
    
    def get_random_action(self):
        return np.random.randint(0, self.num_actions)

    def reset(self):
        self.__init__()
        return self.state

    def serialize_state(self, state:tuple):
        count = state[0] - 12
        usable_ace = int(state[1])
        dealer_card = state[2] - 1
        return count * 20 + usable_ace * 10 + dealer_card

    def deserialize_state(self, serialized_state:int):
        count = serialized_state // 20
        usable_ace = (serialized_state % 20) // 10
        dealer_card = serialized_state % 10
        return count + 12, bool(usable_ace), dealer_card + 1