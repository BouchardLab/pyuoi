#!/usr/bin/env python3

# test single spource radopacyivity detection

import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 3.0         # activity rate in Hz
dt = 0.01        # time interval between resets (seconds)
T = 1.0         # total measurement duration (seconds)

# Derived values
num_steps = int(T / dt)

# Simulate counts per time step (Poisson distribution)

expected_counts_per_step = r * dt
counts_per_step = np.random.poisson(expected_counts_per_step, num_steps)

# Calculate counting rate per time step
counting_rate_per_step = counts_per_step / dt

# Calculate statistics
average_counts_per_second = np.mean(counting_rate_per_step)
variance_counts_per_second = np.var(counting_rate_per_step, ddof=1)

# Theoretical predictions
predicted_average_rate = r
predicted_variance_rate = r / dt

# Print results (C-style formatting)
print('T=%.1f sec,  dt=%.4f sec,  nStep=%d'%(T,dt,num_steps))
print("average counts per second  t:%.2f, m:%.2f" % (predicted_average_rate,average_counts_per_second))
print("variance of counts per second: t:%.2f  m:%.2f" % (predicted_variance_rate, variance_counts_per_second))



# Plot results
plt.figure(figsize=(10,5))
plt.step(np.arange(num_steps)*dt, counting_rate_per_step, where='mid')
plt.xlabel('Time (s)')
plt.ylabel('Counting rate (Hz)')
plt.title('Simulated Counting Rate per Time Step')
plt.grid(True)
plt.show()
