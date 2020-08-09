# Convex Optimization Final Project
## Authors: Tumas Rackaitis, Wes Patterson, Khang Nyugen, and Daniel Firebanks.
## Description: 
This repo includes:
-- An ETL pipeline to get data on every football player since 1970.
-- 3 Time Series models (LSTM-RNN, Time Series AutoRegressor, and Fourier Transform) to predict fantasy scores.
-- An optimizer to weigh those outputs and identify the best prediction. 
-- A portfolio optimization algorithm that Maximizes the Expected Value of Fantasy points (Return) and minimizes Variance of Fantasy Points (Risk). By applying portfolio optimization methodolgies in Finance, we're able to get a pretty good result.

## Results:
We got an A+ on this project, plus we won enough money in draftkings to buy some burritos.

## Infrastructure:

$$$
\documentclass[12pt]{article}
\usepackage{tikz}

\begin{document}

\begin{center}
\begin{tikzpicture}[scale=0.2]
\tikzstyle{every node}+=[inner sep=0pt]
\draw [black] (8.2,-15) circle (3);
\draw (8.2,-15) node {$Player\mbox{ }Career\mbox{ }Data$};
\draw [black] (8.2,-39.3) circle (3);
\draw (8.2,-39.3) node {$Player\mbox{ }Fantasy\mbox{ }Data$};
\draw [black] (14.8,-27.2) circle (3);
\draw (14.8,-27.2) node {$Validation$};
\draw [black] (30,-13.3) circle (3);
\draw (30,-13.3) node {$Time\mbox{ }Series\mbox{ }Regression$};
\draw [black] (34.3,-25.2) circle (3);
\draw (34.3,-25.2) node {$RNN-LSTM$};
\draw [black] (32.2,-44.7) circle (3);
\draw (32.2,-44.7) node {$Fourier\mbox{ }Transform$};
\draw [black] (51.4,-31.3) circle (3);
\draw (51.4,-31.3) node {$Ensemble\mbox{ }Prediction$};
\draw [black] (67.5,-24) circle (3);
\draw (67.5,-24) node {$Team\mbox{ }Roster\mbox{ }Optimizer$};
\draw [black] (9.64,-36.67) -- (13.36,-29.83);
\fill [black] (13.36,-29.83) -- (12.54,-30.3) -- (13.42,-30.78);
\draw (12.17,-34.44) node [right] {$Y$};
\draw [black] (9.63,-17.64) -- (13.37,-24.56);
\fill [black] (13.37,-24.56) -- (13.43,-23.62) -- (12.55,-24.1);
\draw (10.83,-22.28) node [left] {$X$};
\draw [black] (31.896,-26.988) arc (-58.4972:-109.79077:16.699);
\fill [black] (31.9,-26.99) -- (30.95,-26.98) -- (31.48,-27.83);
\draw (25.31,-30.04) node [below] {$X,Y$};
\draw [black] (17.01,-25.18) -- (27.79,-15.32);
\fill [black] (27.79,-15.32) -- (26.86,-15.5) -- (27.53,-16.23);
\draw (24.61,-20.74) node [below] {$X,Y$};
\draw [black] (29.466,-43.469) arc (-117.04415:-153.28419:30.668);
\fill [black] (29.47,-43.47) -- (28.98,-42.66) -- (28.53,-43.55);
\draw (21.14,-39.25) node [left] {$X,Y$};
\draw [black] (32.68,-11.964) arc (110.64118:-10.77701:14.666);
\fill [black] (52.26,-28.43) -- (52.9,-27.74) -- (51.92,-27.55);
\draw (48.94,-13.97) node [above] {$\hat{Y_1}$};
\draw [black] (37.13,-26.21) -- (48.57,-30.29);
\fill [black] (48.57,-30.29) -- (47.99,-29.55) -- (47.65,-30.49);
\draw (41.34,-28.8) node [below] {$\hat{Y_2}$};
\draw [black] (34.66,-42.98) -- (48.94,-33.02);
\fill [black] (48.94,-33.02) -- (48,-33.06) -- (48.57,-33.88);
\draw (43.45,-38.5) node [below] {$\hat{Y_3}$};
\draw [black] (69.01,-26.577) arc (20.99731:-152.2167:9.166);
\fill [black] (69.01,-26.58) -- (68.83,-27.5) -- (69.76,-27.14);
\draw (67.18,-38.74) node [below] {$\hat{Y_opt}$};
\end{tikzpicture}
\end{center}

\end{document}
$$$
