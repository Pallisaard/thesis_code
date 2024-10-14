# Description of SSI

## Main formula

$$
\text{SSI}(x, y) = [l(x, y)]^\alpha \cdot [c(x, y)]^\beta \cdot [s(x, y)]^\gamma
$$

Where

$$
\begin{gather}
l(x, y) = \frac{2 \mu_x \mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1} \\
c(x, y) = \frac{2 \sigma_x \sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2} \\
s(x, y) = \frac{\sigma_{xy} + C_3}{\sigma_x \sigma_y + C_3}
\end{gather}
$$

Here $\mu_x$ and $\mu_y$ are the mean voxel values of the 3D volumes $x$ and
$y$ within the local window, $\sigma_x$ and $\sigma_y$ are standard deviations
of voxel values of the 3D volumes $x$ and $y$ within the local window, and
$\sigma_{xy}$ is the covariance of the 3D volumes $x$ and $y$ within the local
window. $x$ and $y$ can be thought of as 3D windows of the MRI you use to
compute the SSI. Once an SSI has been computed for each window, the aggregate
SSI is computed by a mean over SSI scores for each window:

$$
\text{mSSI} = \frac{\sum_{i=1}^n /text{SSI}_i}{n}
$$

## Computationally friendly rewriting

By setting $\alpha,\beta,\gamma=1$ we compute the SSI by

$$
\text{SSI}(x, y)
= l(x, y) \cdot c(x, y) \cdot s(x, y)
= \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$
