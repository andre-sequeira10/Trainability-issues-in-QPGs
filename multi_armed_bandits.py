import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import torch

def create_circuit(n_qubits,n_layers=None,circ = "simplified_two_design",fim=False, shots=None):

    dev = qml.device("lightning.qubit", wires=n_qubits, shots=shots)

    def RZRY(params):
        #qml.SpecialUnitary(params, wires=range(n_qubits))
        #qml.SimplifiedTwoDesign(initial_layer_weights=init_params, weights=params, wires=range(n_qubits))
        #qml.AngleEmbedding(params,wires=range(n_qubits))
        for q in range(n_qubits):
            qml.Hadamard(wires=q)

        for w in range(n_layers): 
            for q in range(n_qubits):
                index = w * (2*n_qubits) + q * 2
                qml.RZ(params[index],wires=q)
                qml.RY(params[index + 1],wires=q)
        
        qml.broadcast(qml.CNOT , wires=range(n_qubits), pattern="all_to_all")
        
        return qml.probs(wires=range(n_qubits))

    def S2D(init_params,params,measurement_qubits=0,prod_approx=False):
        #qml.SpecialUnitary(params, wires=range(n_qubits))
        qml.SimplifiedTwoDesign(initial_layer_weights=init_params, weights=params, wires=range(n_qubits))
        
        #qml.broadcast(qml.CNOT , wires=range(n_qubits), pattern="all_to_all")
        if not prod_approx:
            return qml.probs(wires=range(measurement_qubits))
        else:
            return [qml.probs(i) for i in range(measurement_qubits)]

    def SU(params):
        qml.SpecialUnitary(params, wires=range(n_qubits))
        
        ZZ = qml.operation.Tensor(qml.PauliZ(0), qml.PauliZ(1))
        for i in range(2,n_qubits):
            ZZ = qml.operation.Tensor(ZZ, qml.PauliZ(i))

        return qml.expval(ZZ)
    
    def simmpleRZRY(params,cnots=True,measurement_qubits=0,sample=False):
        qml.broadcast(qml.Hadamard, wires=range(n_qubits), pattern="single")
        

        qml.broadcast(qml.RZ, wires=range(n_qubits), pattern="single", parameters=params[0])
        qml.broadcast(qml.RY, wires=range(n_qubits), pattern="single", parameters=params[1])

        qml.broadcast(qml.CZ, wires=range(n_qubits), pattern="all_to_all")
 
        return qml.probs(wires=range(measurement_qubits))
        
    def RY(params,y=True,probs=False,prod=False, entanglement=None,measurement_qubits=0):
        qml.broadcast(qml.Hadamard, wires=range(n_qubits), pattern="single")
        qml.broadcast(qml.RY, wires=range(n_qubits), pattern="single", parameters=params)
        #qml.broadcast(qml.CZ, wires=range(n_qubits), pattern="all_to_all")

        if entanglement=="all_to_all":
            qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern="all_to_all")
        
        if y==True:
            #YY = qml.operation.Tensor(qml.PauliY(0), qml.PauliY(1))
            YY = [qml.PauliZ(0), qml.PauliZ(1)]
            for i in range(2,n_qubits):
                #YY = qml.operation.Tensor(YY, qml.PauliY(i))
                YY.append(qml.PauliZ(i))
            
            #return [qml.expval(i) for i in YY]
            return qml.expval(YY)

        elif probs==False:

            ZZ = qml.operation.Tensor(qml.PauliZ(0), qml.PauliZ(1))
            #ZZ = [qml.PauliZ(0), qml.PauliZ(1)]
            for i in range(2,n_qubits):
                ZZ = qml.operation.Tensor(ZZ, qml.PauliZ(i))        
                #ZZ.append(qml.PauliZ(i))        

            #return [qml.expval(i) for i in ZZ]
            return qml.expval(ZZ)

        else:
            if prod:
                return [qml.probs(i) for i in range(n_qubits)]
            else:
                return qml.probs(wires=range(measurement_qubits))
        
    def GHZ(params,measurement_qubits=0):
        qml.RY(params,wires=0)
        qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern="chain")

        return qml.probs(wires=range(measurement_qubits))

    def random_product_state(params,gate_sequence=None):
                
        for i in range(n_qubits):
            qml.RY(np.pi / 4, wires=i)

        for ll in range(len(params)):

            for i in range(n_qubits):
                gate_sequence["{}{}".format(ll,i)](params[ll][i], wires=i)

            #for i in range(n_qubits - 1):
                #qml.CZ(wires=[i, i + 1])
    def SEL(params, measurement_qubits=0):
        qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
        return qml.probs(wires=range(measurement_qubits))
    
    if circ == "rzry":
        qcircuit = RZRY
    elif circ == "simplified_two_design":
        qcircuit = S2D
    elif circ == "special_unitary":
        qcircuit = SU
    elif circ == "simpleRZRY":
        qcircuit = simmpleRZRY
    elif circ == "RY":
        qcircuit = RY
    elif circ == "ghz":
        qcircuit = GHZ
    elif circ == "random_product_state":
        qcircuit = random_product_state
    elif circ == "SEL":
        qcircuit = SEL
    if not fim:
        circuit = qml.QNode(qcircuit, dev,interface="torch", diff_method="best")
    else:
        circuit = qml.QNode(qcircuit, dev)

    return circuit


def policy(probs, policy_type="contiguous-like", n_actions=2):

    if policy_type == "contiguous-like":
        return probs
    elif policy_type == "parity-like":
        policy = torch.zeros(n_actions)
        for i in range(len(probs)):
            a=[]
            for m in range(int(np.log2(n_actions))):
                if m==0:
                    bitstring = np.binary_repr(i,width=n_actions)
                else:
                    bitstring = np.binary_repr(i,width=n_actions)[:-m]
                
                a.append(bitstring.count("1") % 2)
            policy[int("".join(str(x) for x in a),2)] += probs[i]

        return policy    



EPISODES = 100
GAMMA = 0.95
LEARNING_RATE = 0.1
N_ACTIONS = 2**12
TRIALS = 10
POLICIES = ["parity-like"]

# policy gradient algorithm to solve contextual bandit using simplified two design ansatz as the parameterized policy 
# Main part of the code
n_qubits = 16
n_layers = n_qubits-2

best_arm = (2**12)-1

shapes = qml.SimplifiedTwoDesign.shape(n_layers=n_layers, n_wires=n_qubits)
weights =  [np.random.uniform(-np.pi,np.pi,size=shape) for shape in shapes]
#weights_tensor = torch.tensor(weights[1], requires_grad=True)
weights_tensor = torch.tensor(np.random.uniform(-np.pi,np.pi,size=(n_qubits)), requires_grad=True)
#weights_tensor_b = torch.tensor(np.random.uniform(-np.pi,np.pi,size=(n_qubits)), requires_grad=True)
#qc = create_circuit(n_qubits,n_layers=n_layers,circ="simplified_two_design",fim=False)
qc = create_circuit(n_qubits,n_layers=1,circ="RY",fim=False, shots=4096)

opt = torch.optim.Adam([weights_tensor], lr=LEARNING_RATE, amsgrad=True)

for p in POLICIES:
    
    for t in range(TRIALS):
        weights_tensor = torch.tensor(np.random.uniform(-np.pi,np.pi,size=(2,n_qubits)), requires_grad=True)
        #weights_tensor = torch.tensor(np.zeros((2,n_qubits)), requires_grad=True)
        qc = create_circuit(n_qubits,n_layers=1,circ="simpleRZRY",fim=False, shots=4096)

        opt = torch.optim.Adam([weights_tensor], lr=LEARNING_RATE, amsgrad=True)

        total_rewards = []
        best_arms = []
        probs_best_a = []
        gradient_norm= [] 
        for ep in range(EPISODES):
            # Initialize the state
            opt.zero_grad()
            #for s in range(1000):
            #state_ = np.random.randint(num_states)
            #state = list(map(int,np.binary_repr(s,width=n_qubits)))
            #state = np.array(state)
            #state_tensor = torch.tensor(state, requires_grad=False)
            #state_tensor = torch.tensor(state, requires_grad=False)

            p_type = p
            #p_type = "contiguous-like"
            if p_type == "parity-like":
                measure_qubits = n_qubits
                prb = qc(weights_tensor, measurement_qubits=measure_qubits,sample=True)


            elif p_type == "contiguous-like":
                measure_qubits = int(np.log2(N_ACTIONS))
                prb = qc(weights_tensor, measurement_qubits=measure_qubits)

            
            policy_ = policy(prb, policy_type=p_type, n_actions=N_ACTIONS)
            #if p_type == "contiguous-like":
                #policy_ = torch.clamp(policy_,1/n_qubits,1)            
            policy__ = torch.clone(policy_).detach().numpy()

            best_a = np.argmax(policy__)
            prob_best_a = policy__[best_arm]

            dist = torch.distributions.Categorical(probs=policy_)
            
            cost = 0
            for i in range(50):
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action_ = action.item()

                #log_prob = torch.log(policy_[best_arm])
                cost -= action_*log_prob
            cost /= 50  
            #cost = -dist.log_prob(torch.tensor(best_arm))      
            # Update the probs\

            cost.backward()
            opt.step()
            #weights_tensor.data.add_(-LEARNING_RATE * weights_tensor.grad)
            #weights_tensor.grad.zero_()

            #total_rewards.append(reward)
            best_arms.append(best_a)
            probs_best_a.append(prob_best_a)
            #save gradient norm
            gradient_norm.append(np.linalg.norm(weights_tensor.grad.detach().numpy(),2))

            print("Episode: {}, best arm {} || prob best arm: {} || best_arm_measured: {} || gradient norm: {} || policy {}".format(ep+1, best_arm, probs_best_a[-1] ,best_a, gradient_norm[-1],policy__))

            # Update the policy
        #np.save("total_rewards_{}_{}_{}.npy".format(p,n_qubits,t),total_rewards)
        np.save("polylog_gradient_norm_all_to_all_{}_{}_{}.npy".format(p,n_qubits,t),gradient_norm)
        np.save("polylog_weights_all_to_all_{}_{}_{}.npy".format(p,n_qubits,t),weights_tensor.detach().numpy())
        np.save("polylog_best_arms_all_to_all_{}_{}_{}.npy".format(p,n_qubits,t),best_arms)
        np.save("polylog_probs_best_a_all_to_all_{}_{}_{}.npy".format(p,n_qubits,t),probs_best_a)
