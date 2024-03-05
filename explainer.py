import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import hmean
from sklearn.metrics import pairwise_distances


class Explainer:

    def __init__(self, tsne, features):
        self.X = tsne.X
        self.targets = tsne.targets
        self.features = features
        self.Y = tsne.Y
        self.P = tsne.P
        self.Q = tsne.Q
        self.sigma = tsne.sigma
        self.gradients = None
        self.scaled_gradients = None
        self.angles = None

    def compute_all_gradients(self):
        if self.gradients is None:
            self.gradients = []
            for i in range(self.X.shape[0]):
                self.gradients.append(self.compute_gradients(i))
            
            self.gradients = np.array(self.gradients)
        
        return self.gradients
    
    def compute_all_angles(self):
        if self.gradients is None:
            print("Gradients not computed. Please call compute_all_gradients() before.")
        else:
            if self.angles is None:
                all_angles = []

                for i in range(self.X.shape[0]):
                    
                    angles = []
                    for j in range(self.X.shape[0]):

                        angle_1 = self._get_angle(self.gradients[i][0], self.gradients[j][0])
                        angle_2 = self._get_angle(self.gradients[i][1], self.gradients[j][1])

                        angles.append(hmean([angle_1, angle_2]))
                    all_angles.append(angles)
                
                self.angles = np.array(all_angles)
            return self.angles
            
    def _get_angle(self, v1, v2):
        v1_unit = v1 / np.linalg.norm(v1)
        v2_unit = v2 / np.linalg.norm(v2)

        return np.absolute(np.dot(v1_unit, v2_unit))

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
        y2_derivative = self._compute_y2_derivative(i, self.Y, self.P, self.Q)
        
        yx_derivative = self._compute_xy_derivative(i, self.X, self.Y, self.sigma)
        
        derivative = (-np.linalg.inv(y2_derivative)) @ (yx_derivative.T)

        return derivative
    
    def _compute_y2_derivative(self, i, y, P, Q):
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
        
    def _compute_xy_derivative(self, i, X, y, sigma):
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

        if self.gradients is None:
            print("No gradients to save. Please call compute_all_gradients() before.")
        else:
            np.save(path_file, self.gradients)

    def load_gradients(self, path_file):
        self. gradients = np.load(path_file)
        self.scaled_gradients = []
    
    def scale_gradients(self):
        if self.gradients is None:
            print("No gradients to scale. Please call compute_all_gradients() before.")
        else:
            if self.scaled_gradients is None:
                scaled_gradients = []
                for g in self.gradients:
                    norm = np.linalg.norm(g, axis=0)
                    new_g = (g / np.sum(norm))*100
                    scaled_gradients.append(new_g)

                self.scaled_gradients = np.array(scaled_gradients)

    def plot_instance_explanation(self, instance_id):
        if self.gradients is None:
            print("No gradients to plot. Please call compute_all_gradients() before.")
        else:
            fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.32)

            fig.add_trace(go.Bar(x=self.gradients[instance_id][0], y=self.features, orientation="h", name="X-axis gradients", showlegend=False), row=1, col=1)
            fig.add_trace(go.Bar(x=self.gradients[instance_id][1], y=self.features, orientation="h", name="Y-axis gradients", showlegend=False), row=1, col=2)

            fig.update_yaxes(categoryorder="total ascending", showline=True, linewidth=2, linecolor='black', row=1, col=1, mirror=True)
            fig.update_yaxes(categoryorder="total ascending", showline=True, linewidth=2, linecolor='black', row=1, col=2, mirror=True)

            fig.update_xaxes(ticks="outside", showline=True, linewidth=2, linecolor='black', showgrid=False, row=1, col=1, zerolinecolor="grey", zerolinewidth=1, mirror=True)
            fig.update_xaxes(ticks="outside", showline=True, linewidth=2, linecolor='black', showgrid=False, row=1, col=2, zerolinecolor="grey", zerolinewidth=1, mirror=True)

            fig.update_layout(height=1000, width=900, font=dict(size=15), template="simple_white")
            fig.show()

    def plot_scope_explanation(self, instance_id):
        if self.angles is None:
            self.compute_all_angles()
        
        df = pd.DataFrame()
        df["comp-1"] = self.Y[:,0]
        df["comp-2"] = self.Y[:,1]
        df["countries"] = self.targets

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df["comp-1"], y=df["comp-2"],mode="markers", marker=dict(color="black"), marker_size=7, showlegend=False))
        fig.add_trace(go.Scatter(x=[df["comp-1"][instance_id]], y=[df["comp-2"][instance_id]], marker=dict(color="crimson"), marker_size=12, showlegend=False))    
        fig.add_trace(go.Contour(x=df["comp-1"],y=df["comp-2"],z=self.angles[instance_id]))

        fig.update_layout(
            font=dict(size=20),
            autosize=False,
            width=800,
            height=800,
            xaxis=dict(ticks="outside", showgrid=False, zeroline=False, mirror=True),
            yaxis=dict(ticks="outside", showgrid=False, zeroline=False, mirror=True), 
            template="simple_white"
        )

        fig.show()

    def plot_feature_importance_ranking(self):
        
        if self.gradients is None:
            self.compute_all_gradients()
        
        norms = np.zeros(self.gradients.shape[2])

        for g in self.gradients:
            norm = np.linalg.norm(g, axis=0)
            norms += (norm / np.sum(norm))

        mean_norms = norms/self.gradients.shape[2]

        fig = go.Figure()

        fig.add_trace(go.Bar(x=mean_norms, y=self.features, orientation="h", showlegend=False))

        fig.update_yaxes(categoryorder="total ascending", showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_xaxes(ticks="outside", showline=True, linewidth=2, linecolor='black', showgrid=False, zerolinecolor="grey", zerolinewidth=1, mirror=True)

        fig.update_layout( 
                    font=dict(size=15),
                    width=900,
                    height=1000,
                    template="simple_white")

        fig.show()

    def plot_arrow_fields(self, feature_id, scale = 1):

        if self.scaled_gradients is None:
            self.scale_gradients()

        fig = ff.create_quiver(self.Y[:, 0], self.Y[:, 1], self.scaled_gradients[:, 0, feature_id], self.scaled_gradients[:, 1, feature_id], scale=scale)

        # fig.add_trace(go.Contour(x=df["comp-1"],y=df["comp-2"],z=np.array(activations[:, i])))

        fig.update_layout(
            title= "Vector field for " + self.features[feature_id],
            font=dict(size=10),
            xaxis=dict(showgrid=False, zeroline=False, mirror=True),
            yaxis=dict(showgrid=False, zeroline=False, mirror=True),
            height=600,
            width=600,
            showlegend=False,
            template="simple_white"
        )

        fig.show()


    def plot_one_arrow_with_contour(self, feature_id, instance_id, scale = 1):
        if self.scaled_gradients is None:
            self.scale_gradients()
            
        df = pd.DataFrame()
        df["comp-1"] = self.Y[:,0]
        df["comp-2"] = self.Y[:,1]
        df["countries"] = self.targets

        fig = ff.create_quiver([self.Y[instance_id, 0]], [self.Y[instance_id, 1]], [self.scaled_gradients[instance_id, 0, feature_id]], [self.scaled_gradients[instance_id, 1, feature_id]], scale=scale, line=dict(width=3, color="#00f514"))

        fig.add_trace(go.Scatter(x=[df["comp-1"][instance_id]], y=[df["comp-2"][instance_id]], marker=dict(color="crimson"), marker_size=12, showlegend=False))
        fig.add_trace(go.Contour(x=df["comp-1"],y=df["comp-2"],z=np.array(self.X[:, feature_id])))

        fig.update_layout(
            title= "Vector for feature " + self.features[feature_id] + " of " + self.targets[instance_id],
            font=dict(size=10),
            xaxis=dict(showgrid=False, zeroline=False, mirror=True),
            yaxis=dict(showgrid=False, zeroline=False, mirror=True),
            height=600,
            width=600,
            showlegend=False)

        fig.show()

    def plot_combined_gradients(self, instance_id):
        if self.gradients is None:
            print("No gradients to plot. Please call compute_all_gradients() before.")
        else:
            combined_magnitude = np.linalg.norm(self.gradients[instance_id], axis=0)

            fig = go.Figure()

            fig.add_trace(go.Bar(x=combined_magnitude, y=self.features, orientation="h", name="Combined Gradients", showlegend=False))

            fig.update_yaxes(categoryorder="total ascending", showline=True, linewidth=2, linecolor='black', mirror=True)
            fig.update_xaxes(ticks="outside", showline=True, linewidth=2, linecolor='black', showgrid=False, zerolinecolor="grey", zerolinewidth=1, mirror=True)

            fig.update_layout(height=1000, width=900, font=dict(size=15), template="simple_white")
            fig.show()

    def plot_top_gradient_vectors(self, instance_id):
        if self.gradients is None:
            print("No gradients to plot. Please call compute_all_gradients() before.")
        else:
            combined_magnitude = np.linalg.norm(self.gradients[instance_id], axis=0)
            top_features_indices = np.argsort(combined_magnitude)[-4:]
            combined_vectors = np.sum(self.gradients[instance_id][:, top_features_indices], axis=0)

            print(combined_vectors.shape)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=self.Y[:, 0],
                y=self.Y[:, 1],
                mode="markers",
                marker=dict(color="black", size=7),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=[self.Y[instance_id, 0]],
                y=[self.Y[instance_id, 1]],
                mode="markers",
                marker=dict(color="crimson", size=12),
                showlegend=False
            ))

            for component in range(2):  # Adjusted to iterate over components
                fig.add_trace(go.Scatter(
                    x=[self.Y[instance_id, 0], self.Y[instance_id, 0] + combined_vectors[0]*10],  # Adjusted indexing
                    y=[self.Y[instance_id, 1], self.Y[instance_id, 1] + combined_vectors[1]*10],  # Adjusted indexing
                    mode="lines",
                    line=dict(color="blue", width=2),
                    showlegend=False
                ))

            fig.update_layout(
                title="Top Gradient Vectors for Instance " + str(instance_id),
                xaxis=dict(ticks="outside", showgrid=False, zeroline=False, mirror=True),
                yaxis=dict(ticks="outside", showgrid=False, zeroline=False, mirror=True),
                height=600,
                width=600,
                template="simple_white"
            )

            fig.show()
