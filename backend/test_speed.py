from analyzer import RefinedHittingOptimizer
opt = RefinedHittingOptimizer(82, 1.83)
trc_path = '/Users/jetthayes/Desktop/OpenCapData_49175001-6742-4bad-86cb-2c2b22e93225/MarkerData/swing_1.trc'
df = opt.load_trc_file(trc_path)
if df is not None:
    metrics = opt.calculate_trc_metrics(df)
    print("Metrics:", metrics)
else:
    print("Failed to load TRC.")
