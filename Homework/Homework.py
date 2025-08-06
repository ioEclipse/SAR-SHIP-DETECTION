import numpy as np
import matplotlib.pyplot as plt
import math


G=6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
central_mass = 5.972e24  # Mass of the Earth in kg
class Object:
  def __init__(self):
    self.x = 7000000  # Initial x position in meters
    self.y = 0.001  # Initial y position in
    self.z = -1.608 # Initial z position in meters
    self.vx = 0.002  # Initial x velocity in m/s
    self.vy = 1310.359  # Initial y velocity in m/s
    self.vz = 7431.412  # Initial z velocity in m/s
    self.ax=0
    self.ay=0
    self.az=0
    self.mass = 1  # Mass of the object in kg
obj=Object()
print(obj.x, obj.y, obj.z, obj.vx, obj.vy, obj.vz)



def get_error(obj1,obj2):
    time=[]
    error=[]
    for i in range(0 , min(len(obj1), len(obj2))):
       
       time.append(obj1[i][0])
       error.append(((obj1[i][1]/1000-obj2[i][1])**2+(obj1[i][2]/1000-obj2[i][2])**2+(obj1[i][3]/1000-obj2[i][3])**2)**(1/2))
    return error , time


def get_gravity(x,y,z):
    distance = (x**2 + y**2 + z**2)**(1/2)
    a= -G*(central_mass/(distance**2))
    return a*x/distance, a*y/distance, a*z/distance
    
def get_next_step(obj, dt):
    ax, ay, az = get_gravity(obj.x, obj.y, obj.z)
    
    dx= obj.vx * dt + 0.5 * ax * dt**2
    dy= obj.vy * dt + 0.5 * ay * dt**2
    dz= obj.vz * dt + 0.5 * az * dt**2


    obj.x += dx
    obj.y += dy
    obj.z += dz

    obj.vx += ax * dt
    obj.vy += ay * dt
    obj.vz += az * dt

    obj.ax, obj.ay, obj.az = ax, ay, az
    return obj

def rk4_step(obj, dt):
    def get_state(o):
        return np.array([o.x, o.y, o.z, o.vx, o.vy, o.vz])

    def set_state(o, state):
        o.x, o.y, o.z, o.vx, o.vy, o.vz = state

    state0 = get_state(obj)

    def deriv(state):
        x, y, z, vx, vy, vz = state
        ax, ay, az = get_gravity(x, y, z)
        return np.array([vx, vy, vz, ax, ay, az])

    k1 = deriv(state0)
    k2 = deriv(state0 + 0.5 * dt * k1)
    k3 = deriv(state0 + 0.5 * dt * k2)
    k4 = deriv(state0 + dt * k3)

    state_new = state0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    set_state(obj, state_new)
    return obj






def simulate_gravity(obj, t, dt,intsteps):
    steps= math.floor(t/dt)
    positions = []
    for i in range(steps):
        for _ in range(intsteps):
            obj = rk4_step(obj, dt/intsteps)
        positions.append((i*dt,obj.x, obj.y, obj.z))
    return positions







def plot_3d_positions(positions,pos2):
    """
    positions: array-like of shape (N, 3), where each row is (x, y, z)
    """
    positions = np.array(positions)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[0, 0+1]/1000, positions[0, 1+1]/1000, positions[0, 2+1]/1000, c='g', marker='s')
    ax.scatter(positions[:, 0+1]/1000, positions[:, 1+1]/1000, positions[:, 2+1]/1000, c='b', marker='o')
    ax.scatter(pos2[:, 0+1], pos2[:, 1+1], pos2[:, 2+1], c='y', marker='s')
    ax.scatter(0, 0, 0, c='r', marker='o')

    box=10000
    ax.scatter(box, 0, 0, c='r', marker='o')
    ax.scatter(0, box, 0, c='r', marker='o')
    ax.scatter(0, 0, box, c='r', marker='o')
    
    ax.scatter(-box, 0, 0, c='r', marker='o')
    ax.scatter(0, -box, 0, c='r', marker='o')
    ax.scatter(0, 0, -box, c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Position Plot')
    plt.show()
pos=simulate_gravity(obj, 3600*24, 10,1)
pos = np.array(pos)




# Replace 'filename.csv' with your actual file path
data = np.loadtxt('Satellite_PVT_GMAT.csv',comments='#', delimiter=',', skiprows=1)
#plot_3d_positions(data)
plot_3d_positions(pos,data)
print(data[0])
err,t=get_error(pos,data)
plt.figure(figsize=(10,5))
plt.plot(t,err, label='Error')
# plt.plot(time, pos[0][0], label='Velocity')


plt.xlabel('Time (s)')
plt.ylabel('Error (km)')
plt.title('ERROR of orbit simulation vs STK')
plt.legend()
plt.grid(True)
plt.show()