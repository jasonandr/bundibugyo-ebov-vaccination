library(EpiNow2)
library(data.table)

# 1. Read incidence data (date, confirm)
cases <- fread("../data/empirical_incidence.csv")
setnames(cases, c("Date", "Cases"), c("date", "confirm"))
cases$date <- as.Date(cases$date)

# 2. Define Generation Time
# Mean = 15.3, SD = 9.3
generation_time <- generation_time_opts(
  dist = Gamma(mean = 15.3, sd = 9.3)
)

# 3. Define Delays (Incubation + Reporting)
# Incubation: Mean = 8.5, SD = 4.0 (approx)
incubation_period <- delay_opts(
  dist = Gamma(mean = 8.5, sd = 4.0)
)
# Reporting: Mean = 4.0, SD = 2.0 (approx)
reporting_delay <- delay_opts(
  dist = Gamma(mean = 4.0, sd = 2.0)
)

# 4. Run EpiNow2
estimates <- epinow(
  data = cases,
  generation_time = generation_time,
  delays = delay_opts(incubation_period, reporting_delay),
  stan = stan_opts(cores = 4, chains = 4, samples = 4000, warmup = 2000, control=list(adapt_delta=0.95)),
  rt = rt_opts(prior = LogNormal(mean = 1.5, sd = 1.0)),
  gp = gp_opts(),
  logs = NULL,
  return_output = TRUE,
  verbose = TRUE
)

# 5. Extract Rt estimates
summ <- summary(estimates, type = "parameters")
rt_estimates <- summ[variable == "R"]

# Export to CSV
fwrite(rt_estimates, "../results/epinow_rt.csv")
print("EpiNow2 execution complete. Rt estimates saved to ../results/epinow_rt.csv")
