import numpy as np
from sklearn.metrics import pairwise_distances

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

class Explainer:

    def __init__(self, tsne):
        self.X = tsne.X
        self.labels = ["feature" + str(i) for i in range(tsne.X.shape[1])]
        self.Y = tsne.Y
        self.P = tsne.P
        self.Q = tsne.Q
        self.sigma = tsne.sigma
        self.gradients = []
        self.scaled_gradients = []

    def set_labels(self):
        pass

    def compute_all_gradients(self):
        
        for i in range(self.X.shape[0]):
            self.gradients.append(self.compute_gradients(i))
        
        self.gradients = np.array(self.gradients)
        return self.gradients

    def compute_gradients(self, i):
        """
        Function that compute the saliency for an image X and output y.

        Parameters:
        -----------
        i: indice of the input to consider
        
        Return:
        -------
        derivative: t-sne "saliency"
        """
        y2_derivative = self.compute_y2_derivative(i, self.Y, self.P, self.Q)
        
        yx_derivative = self.compute_xy_derivative(i, self.X, self.Y, self.sigma)
        
        derivative = (-np.linalg.inv(y2_derivative)) @ (yx_derivative.T)

        return derivative
    
    def _compute_gradients_old(self, i):
        """
        Function that compute the saliency for an image X and output y.

        Parameters:
        -----------
        i: indice of the input to consider
        
        Return:
        -------
        derivative: t-sne "saliency"
        """
        y2_derivative = self.compute_y2_derivative_old(i, self.Y, self.P, self.Q)
        
        yx_derivative = self.compute_xy_derivative_old(i, self.X, self.Y, self.sigma)
        
        derivative = (-np.linalg.inv(y2_derivative)) @ (yx_derivative)

        return derivative


    def _compute_y2_derivative_old(self, i, y, P, Q):
        """
        Function that compute the second derivative of t-sne regarding y_i
        
        Parameters:
        -----------
        i: indice of the input to consider
        y: low-dimensional space embedding
        P: p-values of t-sne
        Q: q-values of t-sne
        
        Return:
        -------
        4*res: derivative regarding y_i
        """
        res = np.zeros((2,2), np.float64)

        distances = 1 + pairwise_distances(y, squared=True)
        S_q = np.sum( 1/distances ) - np.trace(1/distances)
        
        S_q_d = -4 * np.sum( np.array([(y[i] - y[h]) / ( (1 + (np.linalg.norm(y[i]-y[h])**2))**2 ) for h in range(y.shape[0]) if h != i]),  axis=0)
        
        for j in range(y.shape[0]):
        
            if i != j:
                
                E_ij = 1 + (np.linalg.norm(y[i]-y[j])**2)
                E_ij_d = 2 * ( y[i] - y[j] )

                f_ij = P[i,j]-Q[i,j]
                g_ij = y[i] - y[j]

                f_ij_d = (( (E_ij**(-2)) * 2 * (y[i]-y[j]) * S_q ) + ( 1/E_ij * S_q_d )) / S_q**2
                g_ij_d = np.identity(y.shape[1], np.float64)
                
                res += 1/E_ij * ( f_ij_d.reshape(2, 1) @ g_ij.reshape(1, 2) + f_ij*g_ij_d - (f_ij/E_ij)*(E_ij_d.reshape(2, 1) @ g_ij.reshape(1, 2)))    

        return 4*res
    
    def compute_y2_derivative(self, i, y, P, Q):
        """
        Function that compute the second derivative of t-sne regarding y_i
        
        Parameters:
        -----------
        i: indice of the input to consider
        y: low-dimensional space embedding
        P: p-values of t-sne
        Q: q-values of t-sne
        
        Return:
        -------
        4*res: derivative regarding y_i
        """
        n = y.shape[0]
        m = y.shape[1]

        d_ij = y[i] - y
        d_ij_d = np.identity(m, np.float64)
        d_ij_d = np.tile(d_ij_d, (n, 1, 1))
    
        e_ij = 1 + (np.linalg.norm(d_ij, axis=1))**2
        e_ij_d = 2 * d_ij
        E_ij = 1/e_ij

        distances = 1 + pairwise_distances(y, squared=True) # refactor e_ij with this maybe
        S_q = np.sum( 1/distances ) - np.trace(1/distances)
        S_q_d = -4 * np.sum((e_ij**(-2)).reshape(1, n) @ d_ij, axis=0)
        
        v_ij = (P - Q)[i]     
        v_ij_d = ( ( e_ij_d.T * e_ij**(-2) * S_q ) + ( S_q_d.reshape(m, 1) / e_ij.reshape(1, n) ) ) / ( S_q**2 )
        
        term1 = (v_ij_d * E_ij.reshape(1, n)) @ d_ij
        term2 = (e_ij_d.T * (v_ij * E_ij**2).reshape(1, n)) @ d_ij
        term3 = np.delete( ((v_ij * E_ij).reshape(n, 1, 1) * d_ij_d), i, axis=0).sum(axis=0)
        
        return 4 * ( term1 - term2 + term3 ) 
        
    def _compute_xy_derivative_old(self, i, X, y, sigma):
        """
        Function that compute the second derivative of t-sne regarding x_i

        Parameters:
        -----------
        i: indice of the input to consider
        X: instances in high-dimensional space
        y: embedding in low-dimensional space 
        sigma: sigma values found by t-sne with the chosen perplexity 
        
        Return:
        -------
        4*res: derivative regarding x_i
        """
        res = np.zeros((2,X.shape[1]), np.float64)
        sigma = sigma.reshape((X.shape[0],))

        S_pi = np.sum( np.array([ np.exp( -( (np.linalg.norm(X[i]-X[k])**2) / (2*(sigma[i]**2) ))) for k in range(X.shape[0]) if k != i ])) 
        S_pi_d = - np.sum( np.array([ ((X[i]-X[k])/sigma[i]**2) * (np.exp( -( (np.linalg.norm(X[i]-X[k])**2) / (2*(sigma[i]**2)) ))) for k in range(X.shape[0]) if k != i ]) , axis=0)


        for j in range(X.shape[0]): 
            # print(f"Loop on j in second derivative: {j} / {X.shape[0]}")
            if i != j:
                E_ij = 1 + (np.linalg.norm(y[i]-y[j])**2)

                S_pj = np.sum( np.array([ np.exp( -( (np.linalg.norm(X[j]-X[k])**2) / (2*(sigma[j]**2) ))) for k in range(X.shape[0]) if k != j ])) 

                S_pj_d = (X[j] - X[i]) / (sigma[j]**2) * ( np.exp( -( (np.linalg.norm(X[j] - X[i])**2) / (2*(sigma[j]**2)) )))
               
                P_ji_d = -np.exp( -( (np.linalg.norm(X[i]-X[j])**2) / (2*(sigma[i]**2)) )) * ( ((X[i]-X[j]) / (sigma[i]**2)*S_pi) + (S_pi_d) ) / ( S_pi**2 )
                P_ij_d = np.exp( -( (np.linalg.norm(X[j]-X[i])**2) / (2*(sigma[j]**2)) )) * ( ((X[j]-X[i]) / (sigma[j]**2)*S_pj) - (S_pj_d) ) / ( S_pj**2 )
                
                P_d = (1 / (2*X.shape[0])) * (P_ji_d + P_ij_d)

                res += (y[i]-y[j]).reshape(2,1) / E_ij @ P_d.reshape(1,X.shape[1])

        return 4*res
    
    def compute_xy_derivative(self, i, X, y, sigma):
        """
        Function that compute the second derivative of t-sne regarding x_i

        Parameters:
        -----------
        i: indice of the input to consider
        X: instances in high-dimensional space
        y: embedding in low-dimensional space 
        sigma: sigma values found by t-sne with the chosen perplexity 
        
        Return:
        -------
        4*res: derivative regarding x_i
        """
        n = y.shape[0]
        m = y.shape[1]
        
        sigma = sigma.reshape((X.shape[0],))
        
        y_ij = y[i] - y
        x_ij = X[i] - X
        x_ji = X - X[i]
        e_ij = 1 + (np.linalg.norm(y_ij, axis=1))**2
        E_ij = 1/e_ij

        exp_ij = np.exp( -( (np.linalg.norm(x_ij, axis=1)**2) / (2*(sigma[i]**2)) ) )
        exp_ji = np.exp( -( (np.linalg.norm(x_ji, axis=1)**2) / (2*(sigma**2) ) ))

        S_pi = np.delete( exp_ij, i ).sum()

        S_pi_d = (- (x_ij) * (sigma[i]**(-2)) * exp_ij.reshape(n, 1)).sum(axis=0)
        S_pi_d = np.tile(S_pi_d , (n, 1))
       
        S_pj = ((np.exp( - pairwise_distances(X, squared=True) / ( 2*(sigma**2) ) )).sum(axis=0)) - 1 # on enlève la diagonale (quand j = l)
        S_pj_d = (sigma**(-2) * exp_ji).reshape(n,1) * x_ji

        P_ji_d = ( ( -S_pi * x_ij * sigma[i]**(-2) * exp_ij.reshape(n, 1)) - ( exp_ij.reshape(n, 1) * S_pi_d ) ) / S_pi**2
        P_ij_d = ( ( (S_pj * sigma**(-2) * exp_ji).reshape(n,1) * x_ji ) - ( exp_ji.reshape(n,1) * S_pj_d ) ) / (S_pj**2).reshape(n,1)
        
        v_ij_d = (1 / (2*n)) * (P_ji_d + P_ij_d)

        # return 4 * np.delete( ((v_ij_d * E_ij).reshape(n, 1, 1) * y_ij), i, axis=0).sum(axis=0)
        return 4 * ( v_ij_d.T @ ( y_ij * E_ij.reshape(n, 1) ) )
    
    def save_gradients(self, path_file):
        np.save(path_file, self.gradients)

    def load_gradients(self, path_file):
        self. gradients = np.load(path_file)
        self.scale_gradients = []
    
    def scale_gradients(self):
        self.scale_gradients = []
        for g in self.gradients:
            norm = np.linalg.norm(g, axis=0)
            new_g = (g / np.sum(norm))*100
            self.scaled_gradients.append(new_g)

        self.scaled_gradients = np.array(self.scaled_gradients)

    def plot_arrow_fields(self, feature_id, scale):

        if self.scaled_gradients == []:
            self.scale_gradients()

        fig = ff.create_quiver(self.Y[:, 0], self.Y[:, 1], self.scaled_gradients[:, 0, feature_id], self.scaled_gradients[:, 1, feature_id], scale=scale)

        # fig.add_trace(go.Contour(x=df["comp-1"],y=df["comp-2"],z=np.array(activations[:, i])))

        fig.update_layout(
            font=dict(size=20),
            xaxis=dict(showgrid=False, zeroline=False, mirror=True),
            yaxis=dict(showgrid=False, zeroline=False, mirror=True),
            height=600,
            width=600,
            showlegend=False,
            template="simple_white")

        fig.show()

    def plot_instance_explanation(self, instance_id):
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1)

        fig.add_trace(go.Bar(x=self.gradients[instance_id, 0], y=self.labels, orientation="h", name="X-axis gradients", showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=self.gradients[instance_id, 1], y=self.labels, orientation="h", name="Y-axis gradients", showlegend=False), row=2, col=1)

        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=1, col=1)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=2, col=1)

        fig.update_xaxes(ticks="outside", showline=True, linewidth=2, linecolor='black', showgrid=False, row=1, col=1, zerolinecolor="grey", zerolinewidth=1)
        fig.update_xaxes(ticks="outside", showline=True, linewidth=2, linecolor='black', showgrid=False, row=2, col=1, zerolinecolor="grey", zerolinewidth=1)

        fig.update_layout(height=1000, width=1100, font=dict(size=20))
        fig.show()

    def plot_scope_explanation(self):
        pass