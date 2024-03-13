import numpy as np
import pennylane as qml 
import torch

def create_circuit(n_qubits,n_layers=None,circ = "simplified_two_design",fim=False, shots=None):

    dev = qml.device("default.qubit.torch", wires=n_qubits, shots=shots)

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
            return qml.probs(wires=list(range(measurement_qubits)))
        else:
            return [qml.probs(i) for i in range(measurement_qubits)]

    def SU(params):
        qml.SpecialUnitary(params, wires=range(n_qubits))
        
        ZZ = qml.operation.Tensor(qml.PauliZ(0), qml.PauliZ(1))
        for i in range(2,n_qubits):
            ZZ = qml.operation.Tensor(ZZ, qml.PauliZ(i))

        return qml.expval(ZZ)
    
    def simmpleRZRY(params,cnots=True):
        qml.broadcast(qml.Hadamard, wires=range(n_qubits), pattern="single")
        qml.broadcast(qml.RZ, wires=range(n_qubits), pattern="single", parameters=params[0])
        qml.broadcast(qml.RY, wires=range(n_qubits), pattern="single", parameters=params[1])
        if cnots:
            qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern="chain")

            return qml.expval(qml.PauliZ(n_qubits-1))
        else:
            ZZ = qml.operation.Tensor(qml.PauliZ(0), qml.PauliZ(1))
            for i in range(2,n_qubits):
                ZZ = qml.operation.Tensor(ZZ, qml.PauliZ(i))

            return qml.expval(ZZ)
        
    def RY(params,y=True,probs=False,prod=False, entanglement=None):
        #qml.broadcast(qml.Hadamard, wires=range(n_qubits), pattern="single")
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
                return qml.probs(wires=range(n_qubits))
        
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
    
    def RL(params, measurement_qubits=0):
        qml.RandomLayers(params, ratio_imprim=0.8 ,imprimitive=qml.CZ, wires=range(n_qubits))
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
    elif circ == "RL":
        qcircuit = RL
    if not fim:
        circuit = qml.QNode(qcircuit, dev,interface="torch", diff_method="backprop")
    else:
        circuit = qml.QNode(qcircuit, dev)

    return circuit

def compute_gradient(log_prob, w):
    """Compute gradient of the log probability with respect to weights.
    
    Args:
    - log_prob (torch.Tensor): The log probability tensor.
    - w (torch.Tensor): The weights tensor, with requires_grad=True.

    Returns:
    - numpy.ndarray: The gradient of log_prob with respect to w, flattened.
    """
    if w.grad is not None:
        w.grad.zero_()
    log_prob.backward(retain_graph=True)
    
    if w.grad is None:
        raise ValueError("The gradient for the given log_prob with respect to w is None.")
    
    return w.grad.view(-1).detach().numpy()

def policy(probs, policy_type="contiguous-like", n_actions=2, n_qubits=1):

    if policy_type == "contiguous-like":
        return probs
    elif policy_type == "parity-like":
        policy = torch.zeros(n_actions)
        for i in range(len(probs)):
            a=[]
            for m in range(int(np.log2(n_actions))):
                if m==0:    
                    bitstring = np.binary_repr(i,width=n_qubits)
                else:
                    bitstring = np.binary_repr(i,width=n_qubits)[:-m]
                
                a.append(bitstring.count("1") % 2)
            policy[int("".join(str(x) for x in a),2)] += probs[i]

        return policy    
    
def compute_policy_and_gradient(args):
    n_qubits, shapes, type , n_actions, policy_type, clamp = args

    if policy_type == "parity-like":
        measure_qubits = n_qubits
    else:
        measure_qubits = int(np.log2(n_actions))

    qc = create_circuit(n_qubits, circ=type, fim=False, shots=None)

    if type == "simplified_two_design":
        weights = [np.random.uniform(-np.pi,np.pi,size=shape) for shape in shapes]    
        weights_tensor_init = torch.tensor(weights[0], requires_grad=False)
        weights_tensor_params = torch.tensor(weights[1], requires_grad=True)
        
        probs = qc(weights_tensor_init,weights_tensor_params, measurement_qubits=measure_qubits)

    else:
        weights = [np.random.uniform(-np.pi,np.pi,size=shape) for shape in shapes]    
        weights_tensor_params = torch.tensor(weights, requires_grad=True)

        probs = qc(weights_tensor_params, measurement_qubits=measure_qubits)

    pi = policy(probs, policy_type=policy_type, n_actions=n_actions, n_qubits=n_qubits)

    if clamp is not None:
        pi = torch.clamp(pi, clamp, 1)

    dist = torch.distributions.Categorical(probs=pi)
    
    action = dist.sample()
    log_prob = dist.log_prob(action)

    gradient_no_clamp = np.linalg.norm(compute_gradient(log_prob, weights_tensor_params), 2)
    return gradient_no_clamp


