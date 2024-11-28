path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"

import os
import numpy as np
from environment import environment
from CPG import generator
clear = lambda: os.system('clear')
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
import threading
import time

class SineWaveVisualizer:
    def __init__(self):
        # Initial parameter values for the sine functions
        self.a1, self.h1, self.b1, self.k1 = 1.0, 0.0, 2.0, 0.0  # Parameters for the first sine wave
        self.a2, self.h2, self.b2, self.k2 = 1.0, 0.0, 2.0, 0.0  # Parameters for the second sine wave
        self.a3, self.h3, self.b3, self.k3 = 1.0, 0.0, 2.0, 0.0  # Parameters for the second sine wave

        # Generate time values
        self.t = torch.linspace(0, 50, 500)  # Adjusted to match x-axis range

        # Lock for thread-safe slider value access
        self.lock = threading.Lock()

        # Initialize the plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(left=0.1, right=0.7, bottom=0.3)  # Leave space on the right and bottom for sliders
        self.line1, = self.ax.plot(
            self.t.numpy(), self.forward(self.a1, self.h1, self.b1, self.k1).numpy(), lw=2, label="Sine Wave 1"
        )
        self.line2, = self.ax.plot(
            self.t.numpy(), self.forward(self.a2, self.h2, self.b2, self.k2).numpy(), lw=2, label="Sine Wave 2"
        )
        self.line3, = self.ax.plot(
            self.t.numpy(), self.forward(self.a3, self.h3, self.b3, self.k3).numpy(), lw=2, label="Sine Wave 2"
        )
        # Add legend
        self.ax.legend(loc="upper right")

        # Set fixed axis limits
        self.ax.set_xlim(0, 50)  # Fixed x-axis range
        self.ax.set_ylim(-5, 5)  # Fixed y-axis range

        # Set up the sliders on the right
        self.sliders = []
        slider_params = [
            ('a1', 0.1, 5.0, self.a1),
            ('h1', -10.0, 10.0, self.h1),
            ('b1', 0.1, 10.0, self.b1),
            ('k1', -5.0, 5.0, self.k1),
            ('a2', 0.1, 5.0, self.a2),
            ('h2', -10.0, 10.0, self.h2),
            ('b2', 0.1, 10.0, self.b2),
            ('k2', -5.0, 5.0, self.k2),
            ('a3', 0.1, 5.0, self.a2),
            ('h3', -10.0, 10.0, self.h2),
            ('b3', 0.1, 10.0, self.b2),
            ('k3', -5.0, 5.0, self.k2)
        ]

        slider_ax_height = 0.03
        slider_ax_spacing = 0.05
        for i, (name, val_min, val_max, val_init) in enumerate(slider_params):
            ax_slider = plt.axes(
                [0.75, 0.9 - i * (slider_ax_height + slider_ax_spacing), 0.2, slider_ax_height],
                facecolor='lightgoldenrodyellow'
            )
            slider = Slider(ax_slider, name, val_min, val_max, valinit=val_init)
            slider.on_changed(self.update_plot)
            self.sliders.append(slider)

        # Add 4 non-functional sliders at the bottom
        self.extra_sliders = []
        for i in range(4):
            ax_slider = plt.axes(
                [0.1, 0.2 - i * (slider_ax_height + slider_ax_spacing), 0.6, slider_ax_height],
                facecolor='lightgoldenrodyellow'
            )
            slider = Slider(ax_slider, f"extra_{i+1}", 0.0, 1.0, valinit=0.5)
            self.extra_sliders.append(slider)

        # Start the background thread
        self.running = True
        self.thread = threading.Thread(target=self.main_program)
        self.thread.start()

        # Show the plot
        plt.show()

        # Stop the thread after the plot is closed
        self.running = False
        self.thread.join()

    def forward(self, a, h, b, k):
        return a * torch.sin((self.t - h) / b) + k

    def update_plot(self, _):
        # Update the line data
        with self.lock:
            y1 = self.forward(self.sliders[0].val, self.sliders[1].val, self.sliders[2].val, self.sliders[3].val)
            y2 = self.forward(self.sliders[4].val, self.sliders[5].val, self.sliders[6].val, self.sliders[7].val)
            y3 = self.forward(self.sliders[8].val, self.sliders[9].val, self.sliders[10].val, self.sliders[11].val)

        self.line1.set_ydata(y1.numpy())
        self.line2.set_ydata(y2.numpy())
        self.line3.set_ydata(y3.numpy())
        self.fig.canvas.draw_idle()

    def main_program(self):
        # Initialize the PyBullet physics engine
        env=environment(True)
        env.reset()
        a=generator()
        while 1:
            #get plot values here and print them
            P1 = self.extra_sliders[0].val
            P2 = self.extra_sliders[1].val
            P3 = self.extra_sliders[2].val

            genotype=[0,P1,P2,P3]+[val.val for val in self.sliders]
            #if sum(genotype)!=sum(a.geno): env.reset()
            
            a.set_genotype(genotype)
            env.step(a,[0,0,0,0,0,0,0,0,0,0,0,0],1)
            clear()
            contact=env.quad.getContact()
            for i in range(len(contact)):
                print("\t-- robot link",contact[i][3],"with force",contact[i][9],"Has fallen",env.quad.hasFallen())
            print("Motor positions:",env.quad.motors)
            print("Feet sensors:",env.quad.getFeet())

# Run the visualizer
visualizer = SineWaveVisualizer()

