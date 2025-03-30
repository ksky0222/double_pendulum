import pandas as pd
import matplotlib.pyplot as plt

mode = 3

if mode==0:
    # CSV 파일 경로
    csv_file = 'plot_data/eval_mean_reward.csv'

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

    plt.title('Evaluate Mean Reward', fontsize=16)
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
elif mode==1:
        # CSV 파일 경로
    csv_file = 'plot_data/ep_mean_reward.csv'

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

    plt.title('Episode Mean Reward', fontsize=16)
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
    csv_file = 'plot_data/actor_loss.csv'

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
    csv_file = 'plot_data/critic_loss.csv'

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