import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

mode = 7

if mode==0:
    # CSV 파일 경로
    csv_file = 'plot_data/sac_eval_mean_reward.csv'

    # CSV 읽기
    df = pd.read_csv(csv_file)

    df = df[df['Step'] <= 30_000_000]
    # Step과 Value 컬럼
    x = df['Step']
    y = df['Value']

    # 이동 평균 스무딩
    window_size = 10
    y_smooth = y.rolling(window=window_size, min_periods=1).mean()

    # 플롯 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color='gray', alpha=0.3, label='Raw')
    plt.plot(x, y_smooth, color='blue', label='Smoothed', linewidth=2)

    plt.title('SAC Evaluate Mean Reward', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Reward', fontsize=14)

    # ✅ y_smooth 범위 기준으로 y축 제한
    buffer = 0.05 * (y_smooth.max() - y_smooth.min())
    plt.ylim(y_smooth.min() - buffer, y_smooth.max() + buffer)
    plt.ylim(-1174965.3453125, 2961142.6265625)

    import matplotlib.ticker as ticker
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 저장
    plt.savefig('Evaluate_Mean_Reward.png')
    plt.show()
elif mode==1:
        # CSV 파일 경로
    csv_file = 'plot_data/sac_ep_mean_reward.csv'

    # CSV 읽기
    df = pd.read_csv(csv_file)
    df = df[df['Step'] <= 30_000_000]
    # Step과 Value 컬럼
    x = df['Step']
    y = df['Value']

    # 이동 평균 스무딩
    window_size = 10
    y_smooth = y.rolling(window=window_size, min_periods=1).mean()

    # 플롯 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color='gray', alpha=0.3, label='Raw')
    plt.plot(x, y_smooth, color='blue', label='Smoothed', linewidth=2)

    plt.title('SAC Episode Mean Reward', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Reward', fontsize=14)

    # ✅ y_smooth 범위 기준으로 y축 제한
    buffer = 0.05 * (y_smooth.max() - y_smooth.min())
    plt.ylim(y_smooth.min() - buffer, y_smooth.max() + buffer)

    import matplotlib.ticker as ticker
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 저장
    plt.savefig('Evaluate_Mean_Reward.png')
    plt.show()
elif mode==2:
    # CSV 파일 경로
    csv_file = 'plot_data/sac_actor_loss.csv'

    # CSV 읽기
    df = pd.read_csv(csv_file)

    # Step과 Value 컬럼
    x = df['Step']
    y = df['Value']

    # 이동 평균 스무딩
    window_size = 10
    y_smooth = y.rolling(window=window_size, min_periods=1).mean()

    # 플롯 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color='gray', alpha=0.3, label='Raw')
    plt.plot(x, y_smooth, color='blue', label='Smoothed', linewidth=2)

    plt.title('Actor Loss', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    # ✅ y_smooth 범위 기준으로 y축 제한
    buffer = 0.05 * (y_smooth.max() - y_smooth.min())
    plt.ylim(y_smooth.min() - buffer, y_smooth.max() + buffer)

    import matplotlib.ticker as ticker
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 저장
    plt.savefig('Actor Loss.png')
    plt.show()
elif mode==3:
    # CSV 파일 경로
    csv_file = 'plot_data/sac_critic_loss.csv'

    # CSV 읽기
    df = pd.read_csv(csv_file)

    # Step과 Value 컬럼
    x = df['Step']
    y = df['Value']

    # 이동 평균 스무딩
    window_size = 10
    y_smooth = y.rolling(window=window_size, min_periods=1).mean()

    # 플롯 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color='gray', alpha=0.3, label='Raw')
    plt.plot(x, y_smooth, color='blue', label='Smoothed', linewidth=2)

    plt.title('Critic Loss', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    # ✅ y_smooth 범위 기준으로 y축 제한
    buffer = 0.05 * (y_smooth.max() - y_smooth.min())
    plt.ylim(y_smooth.min() - buffer, 60000000000.00775)
    print(y_smooth.min() - buffer, 60000000000.00775)
    # plt.ylim(75000000, y_smooth.max() + buffer)


    import matplotlib.ticker as ticker
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 저장
    plt.savefig('Critic Loss.png')
    plt.show()

elif mode==4:
    # CSV 파일 경로
    csv_file = 'plot_data/lips_eval_mean.csv'

    # CSV 읽기
    df = pd.read_csv(csv_file)

    # Step과 Value 컬럼
    x = df['Step']
    y = df['Value']

    # 이동 평균 스무딩
    window_size = 10
    y_smooth = y.rolling(window=window_size, min_periods=1).mean()

    # 플롯 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color='gray', alpha=0.3, label='Raw')
    plt.plot(x, y_smooth, color='red', label='Smoothed', linewidth=2)

    plt.title('LIPS Evaluate Mean Reward', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Reward', fontsize=14)

    # ✅ y_smooth 범위 기준으로 y축 제한
    buffer = 0.05 * (y_smooth.max() - y_smooth.min())
    plt.ylim(y_smooth.min() - buffer, y_smooth.max() + buffer)


    import matplotlib.ticker as ticker
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 저장
    plt.savefig('Evaluate_Mean_Reward.png')
    plt.show()
elif mode==5:
        # CSV 파일 경로
    csv_file = 'plot_data/lips_req_mean.csv'

    # CSV 읽기
    df = pd.read_csv(csv_file)

    # Step과 Value 컬럼
    x = df['Step']
    y = df['Value']

    # 이동 평균 스무딩
    window_size = 10
    y_smooth = y.rolling(window=window_size, min_periods=1).mean()

    # 플롯 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color='gray', alpha=0.3, label='Raw')
    plt.plot(x, y_smooth, color='red', label='Smoothed', linewidth=2)

    plt.title('LIPS Episode Mean Reward', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Reward', fontsize=14)

    # ✅ y_smooth 범위 기준으로 y축 제한
    buffer = 0.05 * (y_smooth.max() - y_smooth.min())
    plt.ylim(y_smooth.min() - buffer, y_smooth.max() + buffer)

    import matplotlib.ticker as ticker
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 저장
    plt.savefig('Evaluate_Mean_Reward.png')
    plt.show()

elif mode==6:
    # LIPS CSV
    df_lips = pd.read_csv('plot_data/lips_req_mean.csv')
    df_lips = df_lips[df_lips['Step'] <= 30_000_000]
    x_lips = df_lips['Step']
    y_lips = df_lips['Value']
    y_lips_smooth = y_lips.rolling(window=10, min_periods=1).mean()

    # SAC CSV
    df_sac = pd.read_csv('plot_data/sac_ep_mean_reward.csv')
    df_sac = df_sac[df_sac['Step'] <= 30_000_000]
    x_sac = df_sac['Step']
    y_sac = df_sac['Value']
    y_sac_smooth = y_sac.rolling(window=10, min_periods=1).mean()

    # Plot
    plt.figure(figsize=(10, 6))

    # SAC
    plt.plot(x_sac, y_sac, color='lightblue', alpha=0.3, label='SAC Raw')
    plt.plot(x_sac, y_sac_smooth, color='blue', label='SAC Smoothed', linewidth=2)

    # LIPS
    plt.plot(x_lips, y_lips, color='lightcoral', alpha=0.3, label='LIPS Raw')
    plt.plot(x_lips, y_lips_smooth, color='red', label='LIPS Smoothed', linewidth=2)

    plt.title('Episode Mean Reward', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Reward', fontsize=14)

    # y축 자동 설정 (두 데이터 범위 고려)
    y_combined = pd.concat([y_lips_smooth, y_sac_smooth])
    buffer = 0.05 * (y_combined.max() - y_combined.min())
    plt.ylim(y_combined.min() - buffer, y_combined.max() + buffer)

    # Format axis
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 저장
    plt.savefig('Compare_Episode_Mean_Reward.png')
    plt.show()

elif mode==7:
    # LIPS CSV
    df_lips = pd.read_csv('plot_data/lips_eval_mean.csv')
    df_lips = df_lips[df_lips['Step'] <= 30_000_000]
    x_lips = df_lips['Step']
    y_lips = df_lips['Value']
    y_lips_smooth = y_lips.rolling(window=10, min_periods=1).mean()

    # SAC CSV
    df_sac = pd.read_csv('plot_data/sac_eval_mean_reward.csv')
    df_sac = df_sac[df_sac['Step'] <= 30_000_000]
    x_sac = df_sac['Step']
    y_sac = df_sac['Value']
    y_sac_smooth = y_sac.rolling(window=10, min_periods=1).mean()

    # Plot
    plt.figure(figsize=(10, 6))

    # SAC
    plt.plot(x_sac, y_sac, color='lightblue', alpha=0.3, label='SAC Raw')
    plt.plot(x_sac, y_sac_smooth, color='blue', label='SAC Smoothed', linewidth=2)

    # LIPS
    plt.plot(x_lips, y_lips, color='lightcoral', alpha=0.3, label='LIPS Raw')
    plt.plot(x_lips, y_lips_smooth, color='red', label='LIPS Smoothed', linewidth=2)

    plt.title('Evaluate Mean Reward', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Reward', fontsize=14)

    # y축 자동 설정 (두 데이터 범위 고려)
    y_combined = pd.concat([y_lips_smooth, y_sac_smooth])
    buffer = 0.05 * (y_combined.max() - y_combined.min())
    plt.ylim(y_combined.min() - buffer, y_combined.max() + buffer)

    # Format axis
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 저장
    plt.savefig('Compare_Evaluation_Mean_Reward.png')
    plt.show()