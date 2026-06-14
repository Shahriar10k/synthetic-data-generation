# VAE vs Koopman VAE — Brief Note

## 1. Base VAE

A **Variational Autoencoder (VAE)** learns how to compress and reconstruct data.

For one time-series window:

```text
x → Encoder → z → Decoder → x_hat
```

Where:

* `x` = original signal window
* `z` = latent representation
* `x_hat` = reconstructed signal

The VAE mainly learns:

```text
Can I reconstruct this signal window well?
```

It does **not strongly learn how one window evolves into the next**.

---

## 2. Limitation of Base VAE for Time Series

For time-series data, order matters:

```text
x_t → x_t+1 → x_t+2
```

But a basic VAE often treats windows like separate samples:

```text
reconstruct x_t
reconstruct x_t+1
reconstruct x_t+2
```

So it may learn the **shape** of the signal, but not the **dynamics**.

---

## 3. Koopman VAE

A **Koopman VAE (KoVAE)** extends the VAE by adding a rule for latent time evolution.

It still reconstructs the signal:

```text
x_t → Encoder → z_t → Decoder → x_hat_t
```

But it also learns:

```text
z_t → Koopman Operator K → predicted z_t+1
```

So the model tries to make:

```text
K z_t ≈ z_t+1
```

This means KoVAE learns:

```text
Can I reconstruct the current signal and predict the next latent state?
```

---

## 4. Main Difference

| Feature            | Base VAE          | Koopman VAE                     |
| ------------------ | ----------------- | ------------------------------- |
| Main goal          | Reconstruction    | Reconstruction + dynamics       |
| Input              | One window `x`    | Sequential windows `x_t, x_t+1` |
| Latent space       | Static latent `z` | Dynamic latent `z_t`            |
| Time awareness     | Weak              | Strong                          |
| Extra component    | None              | Koopman operator `K`            |
| Prediction ability | Not natural       | Built into the model            |

---

## 5. Loss Difference

### Base VAE Loss

```text
Loss = Reconstruction Loss + KL Loss
```

### Koopman VAE Loss

```text
Loss = Reconstruction Loss
     + KL Loss
     + Koopman Latent Prediction Loss
     + Future Signal Prediction Loss
```

---

## 6. Intuition

Base VAE says:

```text
I can compress and rebuild this signal.
```

Koopman VAE says:

```text
I can compress and rebuild this signal,
and I can also learn how the hidden state moves over time.
```
