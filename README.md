# Data Set
a. yfinance
yfinance is a free API in Python for download historical stock price published by Yahoo 
Finance. It allows getting real time data for an exchange traded financial instrument. 
The data feed provided the instrument’s volume and price information.
URL: https://pypi.org/project/yfinance/
b. forxpy
forxpy is a free API in Python for download historical FX rates published by Bank of 
Canada. It allows getting the daily historical FX rate in Canadian base rate and business 
date. The data feed provided a date attribute and the FX rate for 24 currencies.
URL: https://pypi.org/project/forxpy/
# Data Analysis
The Long short-term memory (LSTM) and Gated recurrent units (GUR) models are popular machine 
learning models for stock price prediction. There are many speculation commented GUR is outperform 
LSTM in stock price forecasting by able to produce more accurate prediction results. In this exercise, I 
will perform stock price prediction with both models and compare both results.
Background
LSTM and GRU are two types of recurrent neural networks (RNN). They are widely employed in 
sequential or time-series prediction problems. [2][3 RNN is much like a feedforward neural network 
except it can move in a backward direction. It connects inputs and outputs from the previous time step 
and loops through a layer of recurrent neurons network for neuron computation. [4] Both LSTM and GRU 
are having similar architecture except for how they operate information to be kept or discarded at the
loop operation. In loop operation, LSTM has three gates (input gate, forget fate, output gate) while GRU 
only has two gates (update gate, reset gate). The other difference is LSTM has internal memory present 
while GRU does not. [5]
Stock Price Analysis
To retrieve historical stock price, yfinance python library was employed to download stock prices 
published by Yahoo Finance. In this exercise, I have selected Dow Jones Indices (DJI) from the period of 
2017 to 2022 for analysis. 70% of input data was split as training data, 30% of the input data was resided 
as testing data, and data was normalized with MinMaxScaler. Both LSTM and GRU model was set with 1 
dense layer, 288 nodes, and the use of the Adam Optimizer(Adaptive Moment Estimation) and MSE 
(mean squared error) loss metric. Analysis was started with number of epochs (number of training) = 5, 
the prediction result from GRU was with a significantly better accuracy rate when compared to the result 
from LSTM. In the follow prediction evolution table with epochs = 5, it showed GRU was having better 
MAE, MSE, RMSE, R2, and model accuracy %. Further referring to the DJI indices price prediction vs the 
actual graph populated with epochs = 5, GRU model prediction is having more aligned trend when 
compared to the prediction result from LSTM. During simulation process, both models were identified 
having the least loss ratio at number of epochs = 70, thus this epoch point was treated as the optimal
execution point and both models has re-executed again at this point for comparison. In execution with 
number of epochs = 70, the previous findings of GRU outperformed LSTM in prediction accuracy was still 
valid. Referring to the following prediction evolution table with epochs = 70, it showed GRU was having 
better MAE, MSE, RMSE, R2, and model accuracy % when compared to LSTM but their different has 
become smaller at this optimal execution point. 
![image](https://github.com/kitwong5/lstm_gru_comparision/assets/142315009/4c41bdf2-d8a0-4359-ad35-dbe767b3b6d5)
The follow fitting performance graph can further illustrate LSTM model is having more loss rate when 
compared to the GRU model from epochs 5 to 60. Both models reach to their lowest loss rate at around
epochs 70, however, the GRU model still has lesser loss rate when compared to the LSTM model.
![image](https://github.com/kitwong5/lstm_gru_comparision/assets/142315009/277c88f2-4be1-4d67-bf67-824cbd144d1d)
Referred to the study performs on LSTM and GRU by R.Cahuantzi (2021), their study result showed 
“GRUs outperform LSTM networks on low-complexity sequences analysis while on high-complexity 
sequences LSTMs perform better” [6] The findings from R.Cahuantzi was in line with this exercise’s 
results. This stock price prediction only consist of one input variable is a low-complexity sequence 
analysis and in this exercise it find GRU model prediction was more accurate when compared to the 
LSTM model. Part of the reason could be due to the architecture difference between both models that 
GRU has fewer tensor operations when compared to LSTM able to achieve lesser loss rate.
Foreign Exchange Rate (FX Rate) Analysis
In recent stock market news, it suggested the current raise of US stock price was contributed by the 
strong US currency rate and interest rate. This has brought to the idea be further enhance this stock 
price analysis with other attributes, like currency rate. Both LSTM and GRU models are capable of 
multivariate time series prediction. Stock price and US currency rate can be set as input variables and 
made it as 2 dimensions model for predictions in LSTM and GRU models. However, an article published 
by A. Dautel (2020) suggests only a small number of studies have examined deep learning-based 
forecasting models for the FX market because the prediction result was not effective. [7] Based on the
findings from the above article, perform separate LSTM or GRU prediction on stock price and FX rate
then combine the result later was recommended. 
To understand FX rate movement prediction with LSTM and GRU model, a separate execution on FX rate 
data was performed. forxpy python library was employed to download the FX rate published by the 
Bank of Canada. In this execution, the FX rate for USD/AUD from the period of 2017 to 2022 was 
selected. The optimism number of epochs was found to be at 20. Both LSTM and GRU model was 
executed with number of epochs = 20, the prediction result found that both LSTM and GRU model had 
similar accuracy in USD/AUD FX rate movement prediction. This finding was contradicted by the article 
advised by A. Dautel (2020) that FX currency prediction is highly non-stationary and has predictive 
accuracy concerns with LSTM and GRU models. [7] One possible reason that this exercise is able to 
produce an accurate prediction result could be due to the pair of FX currency selected is not in nonstationary time series movement and is suitable for RNN model prediction. If another currency pair were selected, the prediction result could be different.
![image](https://github.com/kitwong5/lstm_gru_comparision/assets/142315009/2ee7fbb6-e96a-4db9-b06d-075d420143c3)
In conclusion, there are various deep-learning modeling tools available for data analysis. However, not 
all of them are suitable for financial instrument analysis. Different financial instruments or financial data 
(like fundamental data, macroeconomic data, and technical data) might perform differently in different 
kinds of deep learning modeling tools. Users should pay attention to the nature of the input data and 
select proper deep-learning tools for data analysis.
# Reference
[2] Wikipedia Contributors (2023). Gated recurrent unit. [online] Wikipedia. Available at: 
https://en.wikipedia.org/wiki/Gated_recurrent_unit
[3] Wikipedia Contributors (2018). Recurrent neural network. [online] Wikipedia. Available at: 
https://en.wikipedia.org/wiki/Recurrent_neural_network
[4] Aurelien Geron (2018). Neural networks and deep learning - Chapert 4 Recurrent Neural Network.
[online] O.Reilly. Available at: https://www.oreilly.com/library/view/neural-networksand/9781492037354/ch04.html
[5] Mystery Vault (2021). LSTM vs GRU in Recurrent Neural Network: A Comparative Study. [online] 
Analytics India Magazome LTD. Available at: https://analyticsindiamag.com/lstm-vs-gru-in-recurrentneural-network-a-comparative-study/
[6] R. Cahuantzi, X. Chen, S. Guttel (2021). A comparison of LSTM an GRU networks for learning 
symbolic sequences. [online] Cornell Universy. Available at: https://arxiv.org/abs/2107.02248
[7] A. Dautel, W. Hardle, S. Lessmann, H Seow (2020). Forex exchange rate forecasting using deep 
recurrent neural networks. [online] Springer Link. Available at:
https://link.springer.com/article/10.1007/s42521-020-00019-x




