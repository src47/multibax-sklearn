def get_posterior_mean(x, model):
    return np.squeeze(model.predict(x))

def get_posterior_std(x, model, n_posterior_samples):
    return np.squeeze(np.std(model.sample_y(x, n_posterior_samples), axis = -1))
    
def calculate_entropy(x, model, n_posterior_samples):
    posterior_variance = (get_posterior_std(x, model, n_posterior_samples) ** 2)
    entropy = 0.5 * np.log(2 * np.pi * posterior_variance) + 0.5
    return entropy
    
def infoBAX(x_domain, x_train, y_train, model, algorithm, n_posterior_samples = 20):
    
    
    gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 10)
    
    
    gpr_model.fit(x_train, y_train)    
    
    term1 = calculate_entropy(x_domain, gpr_model, n_posterior_samples = n_posterior_samples)
    
    term2 = np.zeros(term1.shape)
        
    posterior_samples = gpr_model.sample_y(x_domain, n_posterior_samples)    
    posterior_mean = get_posterior_mean(x_domain, gpr_model)
    posterior_std = get_posterior_std(x_domain, gpr_model, n_posterior_samples = n_posterior_samples)

    for i in range(n_posterior_samples):
        
        gpr_model_fake = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 10)
        posterior_sample = np.squeeze(posterior_samples[:, i])              
        desired_indices = algorithm.identify_subspace(posterior_sample, x_domain)
        
        if len(desired_indices) != 0:
            desired_x = x_domain[desired_indices]  
            predicted_desired_y = posterior_sample[desired_indices]
            
            fake_x_train = np.vstack((x_train, desired_x))      
            fake_y_train = np.vstack((y_train, np.expand_dims(predicted_desired_y, axis=-1)))
        else:
            fake_x_train = x_train
            fake_y_train = y_train

        gpr_model_fake.fit(fake_x_train, fake_y_train)
        
        
        term2 += calculate_entropy(x_domain, gpr_model_fake, n_posterior_samples = n_posterior_samples)

    return term1 -(1/n_posterior_samples) * term2, posterior_mean, posterior_std, posterior_samples
