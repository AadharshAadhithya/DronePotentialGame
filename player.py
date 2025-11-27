import jax.numpy as np
# we will try to solve the following problem:
# min_x0:tau, u0:tau ||x_tau - xhat_tau||^2_Qtau + sum_{t=0}^{tau-1} ||x_t - xhat_t||^2_Q_t + sum_t=0^{tau-1} ||u_t||^2_R

#s.t x0 = xhat0 
# xt+1 = f(xt, ut) t = 0,1,...,tau-1
#g(xt, ut) <= 0 , t=0,1..Taue

#the above is a joint optimization problem for differnt players, i.e , 
# there are n players, and each player has its own x0, xhat0, x_tau, xhat_tau, u0:tau, uhat0:tau

class Player:
    def __init__(self, xref, f,g,tau,Qi,Qtau, Ri):
        self.xref = xref #ref trejectory over tau time steps, so it has  dxtau elements
        self.Qi = Qi 
        self.Qtau = Qtau 
        self.Ri = Ri 
        self.f = f #dynamics xt+1 = f(xt, ut)
        self.tau = tau
      

    def cost(self,x,u):
        cost_i = 0
        
        #cost for tau
        cost_i  = (x[:,-1] - self.xref[:,-1]).T @ self.Qtau @ (x[:,-1] - self.xref[:,-1])
        
        #cost for intermediate time steps
        for t in range(self.tau-1):
            cost_i += (x[:,t] - self.xref[:,t]).T @ self.Qi @ (x[:,t] - self.xref[:,t])
            cost_i += (u[:,t]).T @ self.Ri @ (u[:,t])
          #  cost_i += (x[:,t+1] - self.f(x[:,t], x[:,t+1])).T @ self.Ri @ (x[:,t+1] - self.f(x[:,t], x[:,t+1]))
        #u cost for tau
        cost_i+= (u[:,self.tau-1]).T @ self.Ri @ (u[:,self.tau-1])
        return  cost_i


class PotientialGame:
    def __init__(self, players,g):
        #players is a list of player objects
        self.players = players
        self.n = len(players)
        #we assume each player has the same number of time steps
        self.tau = players[0].tau
        # ref trajectory is appended for each player
        # for N players, ti wil lbe a vector of length N*dxtau
        self.xref = np.concatenate([player.xref for player in players], axis=0)
        # Qtau are block diagonal matrices, (Qtau_1, Qtau_2, ..., Qtau_N) on the diagnal 
        #dimension: N*dx N*dx
        from jax.scipy.linalg import block_diag
        
        # Qtau are block diagonal matrices, (Qtau_1, Qtau_2, ..., Qtau_N) on the diagnal 
        #dimension: N*dx N*dx
        self.Qtau = block_diag(*[player.Qtau for player in players])
        
        # Qi are block diagonal matrices
        self.Qi = block_diag(*[player.Qi for player in players])
        
        # Ri are block diagonal matrices
        self.Ri = block_diag(*[player.Ri for player in players])

        # dynamic function f, takes in appended x and gives out appende xtp1, 
        # construct f from the dynamics of each player
        def f_appended(x_appended, u_appended):
            # x_appended is (N*d,), u_appended is (N*m,)
            # Split into individual player states and controls
            d = players[0].Qtau.shape[0]
            m = players[0].Ri.shape[0]
            next_states = []
            for i, player in enumerate(players):
                x_i = x_appended[i*d:(i+1)*d]
                u_i = u_appended[i*m:(i+1)*m]
                next_state_i = player.f(x_i, u_i)
                next_states.append(next_state_i)
            return np.concatenate(next_states, axis=0)
        self.f = f_appended

        self.g = g #takes in appended x and u and gives out inequality constraints


    def cost(self,x,u):
        #x is dxtau , u is mxtau

        cost_i = 0 

        #cost for tau
        cost_i  = (x[:,-1] - self.xref[:,-1]).T @ self.Qtau @ (x[:,-1] - self.xref[:,-1])
        
        #cost for intermediate time steps
        for t in range(self.tau-1):
            cost_i += (x[:,t] - self.xref[:,t]).T @ self.Qi @ (x[:,t] - self.xref[:,t])
            cost_i += (u[:,t]).T @ self.Ri @ (u[:,t])
          #  cost_i += (x[:,t+1] - self.f(x[:,t], x[:,t+1])).T @ self.Ri @ (x[:,t+1] - self.f(x[:,t], x[:,t+1]))
        #u cost for tau
        cost_i+= (u[:,self.tau-1]).T @ self.Ri @ (u[:,self.tau-1])
        return  cost_i

    def get_constraints(self,x,u):
        init_constraint = x[:,0] - self.xref[:,0]

        #dynamics constraint 
        dynamics_constraints = []
        inequality_constraints = []
        for t in range(self.tau-1):
            dynamics_constraint = x[:,t+1] - self.f(x[:,t], u[:,t])
            dynamics_constraints.append(dynamics_constraint)
            inequality_constraint = self.g(x[:,t], u[:,t])
            inequality_constraints.append(inequality_constraint)
        inequality_constraints.append(self.g(x[:,-1], u[:,-1]))
        # Stack as columns: dynamics_constraints shape (N*d, tau-1)
        # Each element in list is (N*d,), stack along axis=1 to get (N*d, tau-1)
        dynamics_constraints = np.stack(dynamics_constraints, axis=1)
        # Stack as columns: inequality_constraints shape (num_constraints, tau)
        # Each element in list is (num_constraints,), stack along axis=1 to get (num_constraints, tau)
        inequality_constraints = np.stack(inequality_constraints, axis=1)
        return init_constraint, inequality_constraints, dynamics_constraints

