"""
ES-MDA class implementation with pytorch
"""
import torch
import numpy as np 
class ESMDA: 

    def __init__(self, Na, N_ens, N_obs, ens_m_f,ens_d_f, Cd, alpha,measurment_function,model,surface_indeces ,numsave=2):

        #  ens_m_f is the latent variable from encoder out-put and  ens_d_f is the output of observation model 
        # for each ensamble , which is sampled from the latent space distribution 
        self.Na = Na
        self.N = N_ens
        # self.N_m = N_m
        self.N_obs = N_obs
        self.numsave = numsave
        self.decimal = 1-1/(10**numsave)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ens_m_f = ens_m_f # N_m * N_ens
        self.ens_d_f = ens_d_f # N_obs * N_ens # are predicted ones 
        self.alpha = alpha
        self.Cd = Cd  # covariance of measurments N_obs, N_obs
        self.measurment_function = measurment_function
        self.surface_indeces = surface_indeces
        self.model =model
    
    def predict_output_points(self,Latent_ensambels):
        out_put_surface_nodes_prediction =  self.measurment_function (self.model, Latent_ensambels.t(), self.surface_indeces)
        Ensambel_predictions = out_put_surface_nodes_prediction.view(out_put_surface_nodes_prediction.size(0),-1)
        return Ensambel_predictions
      


    

    def update_step_one_iteration(self,measurment):

        self.mesurment =  measurment #  [N_obs, 1]

        m_m = torch.mean(self.ens_m_f, 1, True)
        d_m = torch.mean(self.ens_d_f, 1, True)

        delta_m_f = self.ens_m_f - m_m
        delta_d_f = self.ens_d_f - d_m


 
        Cdd_ft = torch.matmul(delta_d_f, delta_d_f.t())/(self.N - 1)
        Cmd_ft = torch.matmul(delta_m_f, delta_d_f.t())/(self.N - 1)
        
  
        
        self.mesurment = torch.unsqueeze(self.mesurment,1)
        exp = torch.tile(self.mesurment, ( 1,self.N))
        mean_t = torch.zeros_like(exp)




        mean = 0.0
        stddev = 1.0
        R = torch.linalg.cholesky(self.Cd)
        U = R.t()


        noise_t = torch.tensor(np.random.normal(mean, stddev, mean_t.shape),dtype=torch.float32)




        d_obs_t = torch.add(exp, np.sqrt(self.alpha)*torch.matmul(U, noise_t)).cuda()

        # Analysis step (update the latent variable) update rule of ESMDA
        cdd_t = torch.add(Cdd_ft, self.alpha*self.Cd.cuda())

        #fixed_tf_matrix = tf.cast(cdd_t, tf.float64)          
        #s_t, u_t, vh_t = tf.linalg.svd(fixed_tf_matrix)   # CPU bether
        #v_t = tf.cast(vh_t, tf.float32) 
        #s_t = tf.cast(s_t, tf.float32)
        #u_t = tf.cast(u_t, tf.float32)
        
        u_t, s_t , vh_t = torch.linalg.svd(cdd_t)   # CPU bether
        v_t = vh_t
        
        CC = int(self.N_obs*self.decimal)

        # Calculting the inverse using SVD decomposition 
        zero = torch.tensor(0, dtype=torch.float64)
        where = torch.not_equal(s_t, zero)
        # index_non_zero = torch.where(where)

        
        cc_ = (torch.masked_select(s_t, where)).shape[0]

        diagonal_t = s_t[:cc_]
        u_t = u_t[:, :cc_]
        v_t = v_t[:, :cc_]
        s_rt = torch.diag(torch.pow(diagonal_t, -1))


        # this is the Kalman smoother gain 
        K_t = torch.matmul(Cmd_ft, torch.transpose(v_t,0,1) @ s_rt @ torch.transpose(u_t,0,1))

        # Update ensamble 
        self.ens_m_f = self.ens_m_f + torch.matmul(K_t, torch.subtract(d_obs_t, self.ens_d_f)) 
        return  self.ens_m_f 




    def assimilate(self,measurment):
        # It does the assimilation multi times 
        
        # self.computeR()
        # Allocating memory 
        Ensemble = [None] * self.Na
        Ensemble[0] = self.ens_m_f
        
        Ensemble_d = [None] * self.Na
        Ensemble_d[0] = self.ens_d_f
        with torch.no_grad():
         for step in range(self.Na - 1):
             #print("step",step)
             #print("Memory before:")
             #print(torch.cuda.memory_summary(0))
             if step != 0:

                # Ensemble_d[step] = self.predict_output_points(Ensemble[step]).t()
                #self.ens_d_f = Ensemble_d[step]
                 self.ens_d_f = self.predict_output_points(self.ens_m_f).t()
                
             
            # Ensemble[step + 1] = self.update_step_one_iteration(measurment)
            #self.ens_m_f = Ensemble[step + 1]
             self.ens_m_f = self.update_step_one_iteration(measurment)
            

        
        #Ensemble_d[self.Na-1] = self.predict_output_points(Ensemble[self.Na-1]).t() 
         self.ens_d_f = self.predict_output_points(self.ens_m_f).t()
        
        return self.ens_m_f, self.ens_d_f