import jax

# Ensure 64-bit precision for high-accuracy numerical continuation
jax.config.update("jax_enable_x64", True)
