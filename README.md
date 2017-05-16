# VAE_NBP
Variational Auto-encoder with Non-parametric Bayesian Prior 

Instruction:
    
    use vae.py to train the model.

    For test and analysis you can use DPVAE.r. 

Examples:

    source('DPVAE.r')

    init('model')

    z=sample(10)

    image(display(z)[1,,])

    image(display(z)[2,,])

    image(display(z)[3,,])
    
    image(display(z)[8,,])
    
    z2=reconstruct(z)
    
    image(display(z2)[8,,])
    
    z2=reconstruct(z)
    
    image(display(z2)[8,,])
    
    image(display(z)[8,,])
    
    image(display(z2)[8,,])
    
    z2=reconstruct(z)
    
    image(display(z2)[8,,])
    
    z2=reconstruct(z)
    
    image(display(z2)[8,,])
    
    z2=reconstruct(z)
    
    image(display(z2)[8,,])
    
    z2=reconstruct(z)
    
    image(display(z2)[8,,])
    
    z2=reconstruct(z)
    
    image(display(z2)[8,,])
    
    z2=reconstruct(z)
    
    image(display(z2)[8,,])
