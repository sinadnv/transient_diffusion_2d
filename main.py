import numpy as np
import matplotlib.pyplot as plt
import time

# Set up grid size and time step
N = 10
X = np.linspace(0,1,N)
Y = 1-np.linspace(0,1,N)
h = 1/N
dt = 1e-4
alpha = dt/(h*h)

# Set up the initial values as well as boundary conditions
temp = np.zeros((len(X),len(Y)))
temp[0,:] = 1
# Uncomment this line to have a different initial condition and show the steady state solution is independent of the
# initial values as long as the boundary conditions are not time dependent.
# temp[1:len(X)-1,1:len(Y)-1] = 10

# Temperature at the next time step. Will be overwritten in each loop
temp_new = np.array(temp)

# 3D array to store the temperatures in time. The third dimension is the time. I stack the solution at other time steps
# to this array.
temp_store = temp.reshape((len(X),len(Y),1))

# Set up the errors
error_threshold = 1e-4
error = 1
counter = 0
errors = []

# To calculate the CP time
starttime = time.time()

# Explicitly Calculate the temperatures at each time step. will exit the while loop after it reaches a steady state sol.
while error>error_threshold:
    error = 0
    for i in range(1,len(X)-1):
        for j in range(1,len(Y)-1):
            temp_new[i,j] = temp[i,j] + alpha*(temp[i+1,j]+temp_new[i-1,j]+temp[i,j+1]+temp_new[i,j-1]-4*temp[i,j])
            error += abs(temp_new[i,j]-temp[i,j])

    temp_store = np.dstack((temp_store,temp_new))
    errors.append(error)
    counter += 1
    temp = np.array(temp_new)
    # Display the iteration number and error to ensure the solution is converging.
    if counter%100 == 0:
        print('iteration {n} with residual of {e}'.format(n=counter, e=error))

print('Results converged after {n} iterations and CP time of {t} seconds.'.format(n = counter, t = round(time.time()-starttime,2)))
print('It takes {t} seconds to reach the steady state solution.'.format(t = round(dt*counter,2)))

# Create a meshgrid to use contourf command
[x,y] = np.meshgrid(X,Y)

# To show the temperature change in 8 time steps, from initial value to steady state sol
fig, ax = plt.subplots(2,4)
fig.suptitle('Transient Solution')
for i in np.array(range(8)):
    ax[i%2, i//2].contourf(x,y, temp_store[:,:,i*counter//8],20)
    ax[i%2,i//2].set_title('t = {t}'.format(t = round(dt*i*(counter//8),2)))

# To show the error
plot2 = plt.figure(2)
plt.plot(range(1,len(errors),100),errors[::100])
plt.axhline(error_threshold, color = 'red')
plt.xlabel('Iteration Number')
plt.ylabel('Error per Iteration')
plt.legend(['error','error_threshold'])
plt.show()

