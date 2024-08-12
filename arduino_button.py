import pyfirmata
from pyfirmata import Arduino, util
import time

class PushButtonController():
    def __init__(self, idx_button1=2, idx_button2=3):
        # Specify the port that the Arduino board is connected to
        port = '/dev/ttyACM1'  # Adjust this to your Arduino's port
        try:
            board = Arduino(port)
        except Exception as e:
            print(f'Arduino board access error {e}')
            print('to fix it, consult: https://askubuntu.com/questions/1219498/could-not-open-port-dev-ttyacm0-error-after-every-restart')


        # Setup the digital pins
        self.button1_pin = idx_button1  # The digital pin connected to the button
        self.button2_pin = idx_button2  # The digital pin connected to the button

        # Configure the pins
        board.digital[self.button1_pin].mode = pyfirmata.INPUT
        board.digital[self.button2_pin].mode = pyfirmata.INPUT

        self.board = board

        iterator = util.Iterator(board)
        iterator.start()

    def button_get(self):
        button_state1 = self.board.digital[self.button1_pin].read()
        button_state2 = self.board.digital[self.button2_pin].read()
        return button_state1, button_state2
    
    
if __name__ == '__main__':
    push_button_ctrl = PushButtonController(idx_button1=2, idx_button2=3)
    
    print('running the button scheme')
    while True:
        time.sleep(0.1)

        button_state1, button_state2 = push_button_ctrl.button_get()
        if button_state1 is not None and button_state1 is not False:
            print(f'blue button read {button_state1}')     

        if button_state2 is not None and button_state2 is not False:
            print(f'red button read {button_state2}')        