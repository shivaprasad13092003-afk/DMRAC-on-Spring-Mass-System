from springsystem import SpringSystem
import tensorflow.compat.v1 as tf
from controller import MRAC
from refmodel import refModel
from refLibrary import refSignal
import numpy as np
import matplotlib.pyplot as plt


learningON = 1
sim_endTime = 100
start_state = np.reshape([1,1,1,1],(4,1))
env = SpringSystem(start_state)
ref_env = refModel(start_state)
N = int(sim_endTime/env.timeStep)
ref_cmd = refSignal(N)

def main():

    with tf.Session() as sess:
    
        agent = MRAC(sess, env.A, env.B,4,2,0.02)

        sess.run(tf.global_variables_initializer())

        ref_cmd.stepCMD()
        n_idx = 0

        x1 = [start_state[0]]
        x1_ref = [start_state[0]]
        x2 = [start_state[1]]
        x2_ref = [start_state[1]]
        x3 = [start_state[2]]
        x3_ref = [start_state[2]]
        x4 = [start_state[3]]
        x4_ref = [start_state[3]]        
        
        ref_rec_1 = [0]
        ref_rec_2 = [0]

        for idx in range(0, N):

            adap_cntrl = agent.total_Cntrl(env.state, ref_env.state, ref_cmd.refsignal[n_idx])
            env.applyCntrl(adap_cntrl.T)
            ref_env.stepRefModel(ref_cmd.refsignal[n_idx])
            x1.append(env.state[0])
            x1_ref.append(ref_env.state[0])
            x2.append(env.state[1])
            x2_ref.append(ref_env.state[1])
            x3.append(env.state[2])
            x3_ref.append(ref_env.state[2])
            x4.append(env.state[3])
            x4_ref.append(ref_env.state[3])

            ref_rec_1.append(ref_cmd.refsignal[n_idx][0])
            ref_rec_2.append(ref_cmd.refsignal[n_idx][1])
            n_idx = n_idx+1


    plt.figure(1)
    ax1 = plt.subplot(221)
    plt.plot(x1, color='red', label='$x_{1}(t)$')
    plt.plot(x1_ref, color='black', linestyle='--', label='$x_{1rm}(t)$')
    plt.plot(ref_rec_1,color='blue', label='$r_{1}(t)$')
    plt.plot(ref_rec_2,color='yellow', label='$r_{2}(t)$')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('Position of Cart 1')
    plt.legend()

    ax2=plt.subplot(222)
    plt.plot(x2, color='red', label='$\dot{x}_{1}(t)$')
    plt.plot(x2_ref, color='black', linestyle='--', label='$\dot{x}_{1rm}(t)$')
    plt.plot(ref_rec_1,color='blue', label='$r_{1}(t)$')
    plt.plot(ref_rec_2,color='yellow', label='$r_{2}(t)$')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('Velocity of cart 1')
    plt.legend()

    ax3 = plt.subplot(223)
    plt.plot(x3, color='red', label='$x_{2}(t)$')
    plt.plot(x3_ref, color='black', linestyle='--', label='$x_{2rm}(t)$')
    plt.plot(ref_rec_1,color='blue', label='$r_{1}(t)$')
    plt.plot(ref_rec_2,color='yellow', label='$r_{2}(t)$')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('Position of Cart 2')
    plt.legend()

    ax4=plt.subplot(224)
    plt.plot(x4, color='red', label='$\dot{x}_{2}(t)$')
    plt.plot(x4_ref, color='black', linestyle='--', label='$\dot{x}_{2rm}(t)$')
    plt.plot(ref_rec_1,color='blue', label='$r_{1}(t)$')
    plt.plot(ref_rec_2,color='yellow', label='$r_{2}(t)$')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('Velocity of cart 2')
    plt.legend()
    plt.show()


    true_del_1 = [k[0][0] for k in env.TRUE_DELTA_REC]
    true_del_2 = [k[1][0] for k in env.TRUE_DELTA_REC]
    adap_del_1 = [k[0][0] for k in agent.ADAP_CNTRL_REC]
    adap_del_2 = [k[1][0] for k in agent.ADAP_CNTRL_REC]
    

    plt.figure(2)
    ax5=plt.subplot(211)
    plt.plot(adap_del_1, color='red', label='$\\nu_{ad}$')
    plt.plot(true_del_1, color='black', linestyle='--', label='$\Delta(x)$')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('Uncertainty $\Delta(x_{1})$')
    plt.legend()
    plt.title('Deep-MRAC with $\\nu_{ad}=W^{T}\phi^\sigma_{n}(x)$')


    ax6=plt.subplot(212)
    plt.plot(adap_del_2, color='red', label='$\\nu_{ad}$')
    plt.plot(true_del_2, color='black', linestyle='--', label='$\Delta(x)$')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('Uncertainty $\Delta(x_{2})$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()