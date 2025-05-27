library(forecast)

data <- read.csv("close_prices.csv")

close_prices <- as.numeric(data$close)
cleaned <- close_prices[!is.na(close_prices) & is.finite(close_prices)]
close_ts <- ts(cleaned, frequency = 252) # number of trading days in a year

model <- auto.arima(
  close_ts,
  max.p = 3,
  max.q = 3,
  max.order = 5,
  stepwise = FALSE,
  approximation = FALSE
)

print(summary(model))