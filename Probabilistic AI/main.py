# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:42:29 2022

@author: martigtu@stud.ntnu.no
"""

import numpy as np

from stacked_mnist import StackedMNISTData, DataMode
from verification_net import VerificationNet
from AutoEncoder import AutoEncoder
from VariationalAutoEncoder import VariationalAutoEncoder
from functions import visualize, visualize_encoding, visualize_decoding, \
    find_top_anomalies




if __name__=="__main__":
    # Data
    MNIST             = StackedMNISTData( DataMode.MONO_BINARY_COMPLETE )
    stackedMNIST      = StackedMNISTData( DataMode.COLOR_BINARY_COMPLETE )
    MNIST_anom        = StackedMNISTData( DataMode.MONO_BINARY_MISSING )
    stackedMNIST_anom = StackedMNISTData( DataMode.COLOR_BINARY_MISSING )
    
    # Verifier / Classifier model
    model_verifier = VerificationNet(force_learn=False, file_name="Verification_model")
    # --> Coverage: 100.00%, Predictability: 98.62%, Accuracy: 98.39%
    
    
    
    
    ### The Auto Encoder
    
    ## [AE-BASIC]
    AE = AutoEncoder(latent_dim=2, force_learn=False, file_name="AutoEncoder")
    AE.fit(generator=MNIST, batch_size=256, epochs=10)
    # Try on the test data
    x_test, y_test = MNIST.get_full_data_set(training=False)
    x_test_recon = AE.predict(x_test)
    # Visualize reconstruction results, latent encoding and decoding
    visualize(x_test, x_test_recon, N=12)
    visualize_encoding(AE, x_test, y_test, is_AE=True)
    visualize_decoding(AE, N=20, x_range=(-20,20), y_range=(-20,20))
    
    
    # [AE-GEN]
    n = 10000
    # Generate samples
    z_generated = np.random.uniform(-20, 20, size=2*n).reshape(n,2)
    x_generated = np.array( AE.decode(z_generated) )
    # Visualize the generated samples
    visualize(x_generated, "None", N=12)
    # Quality and coverage as a generator
    pred, _ = model_verifier.check_predictability(data=x_generated, correct_labels=None)
    cov = model_verifier.check_class_coverage(data=x_generated, tolerance=.8)
    print(f"Predictability: {100*pred:.2f}%")
    print(f"Coverage: {100*cov:.2f}%")
    # --> Predictability: 83.91%, Coverage: 90.00%
    
    
    # [AE-ANOM]
    anom_AE = AutoEncoder(latent_dim=2, force_learn=False, file_name="AutoEncoder_missing")
    anom_AE.fit(generator=MNIST_anom, batch_size=256, epochs=10)
    # Plot the top anomalies from the test set
    anom_x_test_recon = anom_AE.predict(x_test)
    indecies = find_top_anomalies(x_test, anom_x_test_recon, k=12)
    visualize(x_test[indecies], anom_x_test_recon[indecies], N=12, random=False)
    
    
    # [AE-STACK]
    # Basic
    AE = AutoEncoder(latent_dim=2, force_learn=False, file_name="AutoEncoder_stacked")
    AE.fit(generator=stackedMNIST, batch_size=256, epochs=4)
    # Try on the test data
    sx_test, sy_test = stackedMNIST.get_full_data_set(training=False)
    sx_test_recon = AE.predict(sx_test)
    # Visualize reconstruction results
    visualize(sx_test, sx_test_recon, N=12)
    # Generator
    n = 10000 # number of generated samples
    z_generated = np.random.uniform(low=-20, high=20, size=6*n).reshape(n,2,3)
    x_generated = np.array( AE.decode(z_generated) )
    # Visualize the generated 
    visualize(x_generated, "None", N=12)
    # Quality and coverage as a generator
    pred, _ = model_verifier.check_predictability(data=x_generated, correct_labels=None, tolerance=.5)
    cov = model_verifier.check_class_coverage(data=x_generated, tolerance=.5)
    print(f"Predictability: {100*pred:.2f}%")
    print(f"Coverage: {100*cov:.2f}%")
    # Anomaly detector
    anom_AE = AutoEncoder(latent_dim=2, force_learn=False, file_name="AutoEncoder_missing")
    anom_AE.fit(generator=stackedMNIST_anom, batch_size=256, epochs=10)
    # Plot the top anomalies from the test set
    anom_sx_test_recon = anom_AE.predict(sx_test)
    indecies = find_top_anomalies(sx_test, anom_sx_test_recon, k=12)
    visualize(sx_test[indecies], anom_sx_test_recon[indecies], N=12, random=False)    
    
    
    
    
    ### Variational Auto Encoder
    
    ## [VAE-BASIC]
    VAE = VariationalAutoEncoder(latent_dim=2, force_learn=False, file_name="VariationalAE")
    VAE.fit(generator=MNIST, batch_size=256, epochs=20)
    # Try on the test data
    x_test, y_test = MNIST.get_full_data_set(training=False)
    x_test_recon = VAE.predict(x_test)
    # Visualize reconstruction results, latent encoding and decoding
    visualize(x_test, x_test_recon, N=12)
    visualize_encoding(VAE, x_test, y_test, is_AE=False)
    visualize_decoding(VAE, N=20, x_range=(-2.5,2.5), y_range=(-2.5,2.5))
    
    
    ## [VAE-GEN]
    n = 10000
    # Generate samples
    z_generated = np.random.uniform(-2.5, 2.5, size=2*n).reshape(n,2)
    x_generated = np.array( VAE.decode(z_generated) )
    # Visualize the generated samples
    visualize(x_generated, "None", N=12)
    # Quality and coverage as a generator
    pred, _ = model_verifier.check_predictability(data=x_generated, correct_labels=None)
    cov = model_verifier.check_class_coverage(data=x_generated, tolerance=.8)
    print(f"Predictability: {100*pred:.2f}%")
    print(f"Coverage: {100*cov:.2f}%")
    # --> Predictability: 86.26%, Coverage: 100.00%
    
    
    ## [VAE-ANOM]
    anom_VAE = VariationalAutoEncoder(latent_dim=2, force_learn=False, file_name = "VariationalAE_missing")
    anom_VAE.fit(generator=MNIST_anom, batch_size=512, epochs=5)
    # Plot the top anomalies from the test set
    anom_x_test_recon = anom_VAE.predict(x_test)
    indecies = find_top_anomalies(x_test, anom_x_test_recon, k=12)
    visualize(x_test[indecies], anom_x_test_recon[indecies], N=12, random=False)
    
    
    ## [VAE-STACK]
    VAE = VariationalAutoEncoder(latent_dim=2, force_learn=False, file_name="VariationalAE_stacked")
    VAE.fit(generator=stackedMNIST, batch_size=256, epochs=4)
    # Try on the test data
    sx_test, sy_test = stackedMNIST.get_full_data_set(training=False)
    sx_test_recon = VAE.predict(sx_test)
    # Visualize reconstruction results
    visualize(sx_test, sx_test_recon, N=12)
    # Generator
    n = 10000 # number of generated samples
    z_generated = np.random.uniform(low=-2.5, high=2.5, size=6*n).reshape(n,2,3)
    x_generated = np.array( VAE.decode(z_generated) )
    # Visualize the generated
    visualize(x_generated, "None", N=12)
    # Quality and coverage as a generator
    pred, _ = model_verifier.check_predictability(data=x_generated, correct_labels=None, tolerance=.5)
    cov = model_verifier.check_class_coverage(data=x_generated, tolerance=.5)
    print(f"Predictability: {100*pred:.2f}%")
    print(f"Coverage: {100*cov:.2f}%")
    # --> Predictability: 90.14%, Coverage: 83.20%
    # Anomaly detector
    anom_VAE = VariationalAutoEncoder(latent_dim=2, force_learn=False, file_name="VariationalAE_missing")
    anom_VAE.fit(generator=stackedMNIST_anom, batch_size=256, epochs=10)
    # Plot the top anomalies from the test set
    anom_sx_test_recon = anom_VAE.predict(sx_test)
    indecies = find_top_anomalies(sx_test, anom_sx_test_recon, k=12)
    visualize(sx_test[indecies], anom_sx_test_recon[indecies], N=12, random=False)


