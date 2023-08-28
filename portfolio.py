import streamlit as st
import datetime
import yfinance as yf
import pandas as pd
import sqlite3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

# Create a connection to the SQLite database
conn = sqlite3.connect('../../OneDrive/Desktop/Acad/Sem 5/DBMS/finapp.db')
c = conn.cursor()

# Set the page title
st.set_page_config(page_title='Stock Portfolio App', page_icon=':chart_with_upwards_trend:')

# Create or get the session state
def get_session_state():
    if 'user_id' not in st.session_state.keys():
        st.session_state.user_id = None
    return st.session_state.user_id

st.title('📈 Stock Portfolio App')
st.write('This application is a powerful tool designed to help users '
         'manage and track their investment portfolios. With real-time stock data retrieval, '
         'users can add stocks to their portfolio, track their performance, and analyze key metrics '
         'such as portfolio value, returns, and more. The app also provides features for buying and selling '
         'stocks, updating portfolio details, and comparing portfolio performance against popular benchmarks.'
         ' With an intuitive interface and comprehensive functionality, the app empowers users to make informed '
         'investment decisions and stay on top of their financial goals.')

with st.sidebar:
    st.header('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        # Check if the username and password match a user in the database
        c.execute('''SELECT id FROM users WHERE username = ? AND password = ?''',
                  (username, password))
        result = c.fetchone()
        if result:
            user_id = result[0]
            st.session_state.user_id = user_id  # Store user_id in session state
            st.success('Login successful!')
        else:
            st.error('Invalid username or password')

    st.header('Register')
    new_username = st.text_input('New username')
    new_password = st.text_input('password', type='password')
    if st.button('Register'):
        # Check if the username already exists in the database
        c.execute('''SELECT id FROM users WHERE username = ?''', (new_username,))
        result = c.fetchone()
        if result:
            st.error('Username already exists')
        else:
            # Add the new user to the database
            c.execute('''INSERT INTO users (username, password) VALUES (?, ?)''',
                      (new_username, new_password))
            conn.commit()
            st.success('Registration successful!')

# Check if user is logged in
if get_session_state() is not None:
    # Define functions to interact with the database
    def add_stock_to_portfolio(user_id, stock_id, shares, purchase_price, purchase_date,current):
        # Insert the portfolio data into the database, including the current stock price
        current_date = datetime.date.today()
        c.execute('''INSERT INTO portfolio (user_id, stock_id, shares, purchase_price, purchase_date, current_price, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (user_id, stock_id, shares, purchase_price, purchase_date, current,current_date))
        conn.commit()


    def update_stock_in_portfolio(portfolio_id, shares, purchase_price, purchase_date):
        c.execute('''UPDATE portfolio SET shares = ?, purchase_price = ?, purchase_date = ?
                        WHERE id = ?''',
                  (shares, purchase_price, purchase_date, portfolio_id))
        conn.commit()

    def get_user_portfolio(user_id):
        c.execute('''SELECT portfolio.id, stocks.ticker, stocks.company_name, stocks.sector, portfolio.shares, portfolio.purchase_price, portfolio.purchase_date, portfolio.current_price, portfolio.last_updated
                        FROM portfolio JOIN stocks ON portfolio.stock_id = stocks.id
                        WHERE portfolio.user_id = ?''',
                  (user_id,))
        return c.fetchall()


    def get_portfolio_value(user_id):
        c.execute('''SELECT SUM((portfolio.current_price - portfolio.purchase_price) * portfolio.shares) AS value
                        FROM portfolio
                        WHERE portfolio.user_id = ?''',
                  (user_id,))
        result = c.fetchone()
        return result[0] if result[0] else 0.0


    def get_benchmark_prices(benchmark_ticker, start_date, end_date):
        # Use yfinance to retrieve historical prices for the benchmark
        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)
        benchmark_prices = benchmark_data['Close'].values.tolist()
        return benchmark_prices


    def get_benchmark_data(benchmark_id):
        c.execute('''SELECT ticker, name, description FROM benchmark WHERE id = ?''', (benchmark_id,))
        result = c.fetchone()
        if result:
            return result
        else:
            return None  # Or handle the case when no data is found in the database


    def get_portfolio_prices(user_id, start_date, end_date):
        c.execute('''SELECT portfolio.purchase_date, portfolio.current_price, portfolio.shares
                        FROM portfolio
                        WHERE portfolio.user_id = ? AND portfolio.purchase_date BETWEEN ? AND ?''',
                  (user_id, start_date, end_date))
        rows = c.fetchall()

        # Calculate the daily portfolio value based on the number of shares and the current price
        portfolio_prices = []
        for row in rows:
            date, price, shares = row
            value = price * shares
            portfolio_prices.append((date, value))

        # Aggregate the daily portfolio values by date
        portfolio_prices_by_date = {}
        for date, value in portfolio_prices:
            if date in portfolio_prices_by_date:
                portfolio_prices_by_date[date] += value
            else:
                portfolio_prices_by_date[date] = value

        # Fill in any missing dates with the previous date's portfolio value
        current_date = start_date
        portfolio_values = []
        while current_date <= end_date:
            if current_date in portfolio_prices_by_date:
                portfolio_values.append(portfolio_prices_by_date[current_date])
            elif portfolio_values:
                portfolio_values.append(portfolio_values[-1])
            else:
                portfolio_values.append(0.0)
            current_date = increment_date(current_date)

        return portfolio_values

    def get_performance_data(user_id, benchmark_id, start_date, end_date):
        # Get benchmark data
        benchmark_ticker, benchmark_name, benchmark_description = get_benchmark_data(benchmark_id)

        # Get benchmark prices
        benchmark_prices = get_benchmark_prices(benchmark_ticker, start_date, end_date)

        # Get portfolio prices
        portfolio_prices = get_portfolio_prices(user_id, start_date, end_date)

        # Calculate portfolio returns and benchmark returns
        portfolio_return = (portfolio_prices[-1] - portfolio_prices[0]) / portfolio_prices[0]
        benchmark_return = (benchmark_prices[-1] - benchmark_prices[0]) / benchmark_prices[0]

        # Calculate excess returns and tracking error
        excess_return = portfolio_return - benchmark_return
        tracking_error = calculate_tracking_error(portfolio_prices, benchmark_prices)

        # Create a dictionary of performance data
        performance_data = {
            'benchmark_ticker': benchmark_ticker,
            'benchmark_name': benchmark_name,
            'benchmark_description': benchmark_description,
            'portfolio_value': portfolio_prices[-1],
            'benchmark_value': benchmark_prices[-1],
            'portfolio_return': portfolio_return,
            'benchmark_return': benchmark_return
        }

        return performance_data

    # Define a function to calculate tracking error
    def calculate_tracking_error(portfolio_prices, benchmark_prices):
        differences = []
        for i in range(len(portfolio_prices)):
            difference = portfolio_prices[i] - benchmark_prices[i]
            differences.append(difference)
        mean_difference = sum(differences) / len(differences)
        squared_differences = [(difference - mean_difference) ** 2 for difference in differences]
        mean_squared_difference = sum(squared_differences) / len(squared_differences)
        return mean_squared_difference ** 0.5

    # Define a function to increment a date by one day
    def increment_date(date_string):
        year, month, day = date_string.split('-')
        next_day = str(int(day) + 1)
        if len(next_day) == 1:
            next_day = '0' + next_day
        return f'{year}-{month}-{next_day}'


    st.markdown("<hr>", unsafe_allow_html=True)
    st.header('🔍Market Data Retrieval and Logarithmic Return Calculation')
    st.markdown("<hr>", unsafe_allow_html=True)
    # Get user input for stock ticker symbol
    ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL):")

    # Define timeframe options
    timeframes = {
        "5D": 5,
        "10D": 10,
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "YTD": "YTD"
    }

    # Get user input for timeframe selection
    selected_timeframe = st.selectbox("Select a timeframe:", list(timeframes.keys()))

    # Placeholder for stock data and log returns
    stock_data = None
    log_returns = None


    # Define function to retrieve stock data
    @st.cache_data(ttl=60 * 5)  # Caches the data for 5 minutes to avoid redundant API calls
    def load_stock_data(ticker, timeframe):
        try:
            if isinstance(timeframe, int):
                data = yf.download(ticker, period=f"{timeframe}d")
            else:
                data = yf.download(ticker, period=timeframe)
            return data
        except Exception as e:
            st.error("Error retrieving stock data. Please check your inputs.")
            st.stop()


    # Retrieve stock data and calculate log returns
    if st.button("Calculate"):
        stock_data = load_stock_data(ticker, timeframes[selected_timeframe])

        # Check if stock data is empty or contains valid records
        if stock_data.empty or stock_data['Close'].isnull().all():
            st.error("No data available for the specified stock ticker and timeframe.")
        else:
            # Calculate logarithmic returns using adjusted close prices
            log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1))

    # Display the stock data and log returns
    if stock_data is not None:
        st.subheader("Stock Data")
        stock_data = stock_data.iloc[::-1]
        st.dataframe(stock_data)

        col1, col2 = st.columns(2)
        # Plot log returns using seaborn
        with col1:
            sns.set_style("darkgrid")
            fig, ax = plt.subplots()
            sns.lineplot(data=log_returns, ax=ax)
            st.pyplot(fig)

        # Display the log returns table
        with col2:
            st.subheader("Log Returns")
            log_returns = log_returns.iloc[::-1]
            st.dataframe(log_returns)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.header('➕ Add a new stock to your portfolio')
    st.markdown("<hr>", unsafe_allow_html=True)
    ticker = st.text_input('Ticker symbol')
    shares = st.number_input('Number of shares', min_value=1, step=1)
    purchase_price = st.number_input('Purchase price', min_value=0.0, step=0.01)
    purchase_date = st.date_input('Purchase date')
    add_stock_button = st.button('Add stock')
    if add_stock_button and ticker and shares and purchase_price and purchase_date:
        # Check if the ticker symbol exists in the database
        c.execute('''SELECT id FROM stocks WHERE ticker = ?''', (ticker,))
        result = c.fetchone()
        stock_id = str(ticker)
        stock_data = yf.download(stock_id, start=purchase_date)
        if stock_data.empty or stock_data['Close'].isnull().all():
            st.error("No data available for the specified stock ticker and date.")
        current_price = stock_data['Close'].iloc[0]
        if result:
            stock_id = result[0]
            add_stock_to_portfolio(st.session_state.user_id, stock_id, shares, purchase_price, str(purchase_date),current_price)
            st.success('Stock added to portfolio!')
        else:
            st.error('Invalid ticker symbol')
    st.markdown("<hr>", unsafe_allow_html=True)
    # st.header('Update a stock in your portfolio')
    # portfolio_data = get_user_portfolio(st.session_state.user_id)
    # if portfolio_data:
    #     portfolio_dropdown_options = [f'{row[1]} ({row[2]})' for row in portfolio_data]
    #     portfolio_dropdown_index = st.selectbox('Select a stock', range(len(portfolio_data)),
    #                                             format_func=lambda i: portfolio_dropdown_options[i])
    #     stock_id = portfolio_data[portfolio_dropdown_index][0]
    #     shares = st.number_input('Number of shares', min_value=1, step=1,
    #                              value=portfolio_data[portfolio_dropdown_index][4])
    #     purchase_price = st.number_input('Purchase price', min_value=0.0, step=0.01,
    #                                      value=portfolio_data[portfolio_dropdown_index][5])
    #
    #     if st.button('Update stock'):
    #         update_stock_in_portfolio(stock_id, shares, purchase_price, str(purchase_date))
    #         st.success('Stock updated in portfolio!')

    def get_user_portfolio2(user_id):
        c.execute('''SELECT portfolio.id, stocks.ticker, stocks.company_name, stocks.sector, 
        portfolio.shares, portfolio.purchase_price, portfolio.purchase_date, portfolio.current_price, 
        portfolio.last_updated,stocks.id
                        FROM portfolio JOIN stocks ON portfolio.stock_id = stocks.id
                        WHERE portfolio.user_id = ?''',
                  (user_id,))
        return c.fetchall()

    st.header('📊 Update a stock in your portfolio')
    st.markdown("<hr>", unsafe_allow_html=True)
    def insert_transaction(user_id, stock_id, transaction_type, shares, price,date ):
        c.execute('''INSERT INTO transactions (user_id, stock_id, transaction_type, shares, price, date)
                      VALUES (?, ?, ?, ?, ?, ?)''',
                  (user_id, stock_id, transaction_type, shares, price, date))
        conn.commit()
    def update_stock_in_portfolio2(stock_id, new_shares, current_purchase_price):
        if new_shares == 0:
            # If new_shares is 0, remove the stock from the portfolio
            c.execute('''DELETE FROM portfolio WHERE stock_id = ?''', (stock_id,))
            conn.commit()
            return

        # Update the stock details in the portfolio
        c.execute('''UPDATE portfolio SET shares = ?, purchase_price = ?, purchase_date = ?
                    WHERE stock_id = ?''', (new_shares, current_purchase_price, purchase_date, stock_id))
        conn.commit()
        st.success('Stock updated in portfolio!')
    portfolio_data = get_user_portfolio2(st.session_state.user_id)
    if portfolio_data:
        portfolio_dropdown_options = [f'{row[1]} ({row[2]})' for row in portfolio_data]
        portfolio_dropdown_index = st.selectbox('Select a stock', range(len(portfolio_data)),
                                                format_func=lambda i: portfolio_dropdown_options[i])
        stock_id = portfolio_data[portfolio_dropdown_index][1]
        nid=portfolio_data[portfolio_dropdown_index][9]
        current_shares = portfolio_data[portfolio_dropdown_index][4]
        current_purchase_price = portfolio_data[portfolio_dropdown_index][5]

        st.write(f"Current shares: {current_shares}")
        st.write(f"Purchase price: {current_purchase_price}")

        new_shares = st.number_input('Number of shares', min_value=0, step=1, value=current_shares)

        if new_shares == current_shares:
            st.warning("No change in shares.")

        elif new_shares < current_shares:
            # Selling shares
            shares_sold = current_shares - new_shares
            selling_date = st.date_input('Selling date')

            if st.button('Sell shares'):
                # Fetch current stock value on the selling date
                selling_date_str = selling_date.strftime('%Y-%m-%d')
                stock_data = yf.download(str(stock_id), start=selling_date_str, end=selling_date_str)
                if stock_data.empty or stock_data['Close'].isnull().all():
                    st.error("No data available for the specified stock ticker and date.")
                else:
                    current_stock_value = stock_data['Close'].iloc[0]

                    # Calculate profit or loss
                    profit_loss = (current_stock_value - current_purchase_price) * shares_sold

                    # Remove stock from portfolio
                    update_stock_in_portfolio2(nid,new_shares, current_purchase_price)

                    # Insert transaction into the transaction table
                    insert_transaction(st.session_state.user_id,stock_id,'SELL',shares_sold, current_stock_value ,selling_date)

                    if profit_loss < 0:
                        st.error(f"Sold {shares_sold} shares for a loss of ${abs(profit_loss):.2f}.")
                    else:
                        st.success(f"Sold {shares_sold} shares for a profit of ${profit_loss:.2f}.")

        else:
            # Buying more shares
            shares_bought = new_shares - current_shares
            buying_price = st.number_input('Buying price', min_value=0.0, step=0.01, value=current_purchase_price)
            buying_date = st.date_input('Buying date')

            if st.button('Buy shares'):
                # Update stock in portfolio
                update_stock_in_portfolio2(nid, new_shares, current_purchase_price)

                # Insert transaction into the transaction table
                insert_transaction(st.session_state.user_id,stock_id,'BUY',new_shares, buying_price ,buying_date)

                st.success(f"Bought {shares_bought} more shares for {stock_id}.")

    st.markdown("<hr>", unsafe_allow_html=True)


    st.header('🧾 View your portfolio holdings')
    portfolio_data = get_user_portfolio(st.session_state.user_id)
    if portfolio_data:
        # Create a DataFrame with the portfolio data
        df_portfolio = pd.DataFrame(portfolio_data,
                                    columns=['ID', 'Ticker', 'Company', 'Sector', 'Shares', 'Purchase Price',
                                             'Purchase Date', 'Current Price', 'Last Updated'])

        # Display the portfolio table
        st.table(df_portfolio)

        portfolio_value = get_portfolio_value(st.session_state.user_id)
        st.write(f'Total portfolio returns: ${portfolio_value:.2f}')
        if st.button('Chart Portfolio Returns'):
            # Add a chart of the portfolio's performance over time
            if len(portfolio_data) > 0:
                def get_earliest_purchase_date(portfolio_data):
                    earliest_date = None
                    for row in portfolio_data:
                        purchase_date = datetime.datetime.strptime(row[6], '%Y-%m-%d').date()
                        if earliest_date is None or purchase_date < earliest_date:
                            earliest_date = purchase_date
                    return earliest_date


                # Function to calculate the portfolio value for each day
                def calculate_portfolio_value2(portfolio_data, start_date, end_date):
                    portfolio_dates = pd.date_range(start=start_date, end=end_date)
                    portfolio_values = []

                    for date in portfolio_dates:
                        date_str = date.strftime('%Y-%m-%d')
                        total_value = 0.0

                        for row in portfolio_data:
                            purchase_date = datetime.datetime.strptime(row[6], '%Y-%m-%d').date()
                            stock_data = yf.download(row[1], start=purchase_date, end=date_str)
                            if not stock_data.empty:
                                current_price = stock_data['Close'].iloc[-1]
                                stock_value = (current_price-purchase_price) * row[4]
                                total_value += stock_value

                        if date==purchase_date:
                            for row in portfolio_data:
                                stock_val=purchase_price * row[4]
                                total_value=total_value+stock_val

                        portfolio_values.append(total_value)
                        for i in range(0,len(portfolio_values)):
                            data=portfolio_values[i]
                            if data==0:
                                portfolio_values[i]=portfolio_values[i-1]
                    return portfolio_dates, portfolio_values


                st.set_option('deprecation.showPyplotGlobalUse', False)
                # Get the earliest purchase date from the portfolio
                earliest_date = get_earliest_purchase_date(portfolio_data)

                # Calculate the portfolio value for each day between the earliest purchase date and current date
                current_date = datetime.date.today()
                portfolio_dates, portfolio_values = calculate_portfolio_value2(portfolio_data, earliest_date, current_date)

                # Set the Seaborn style to darkgrid
                sns.set_style('darkgrid')

                # Create a DataFrame with the portfolio data
                chart_data = pd.DataFrame({'Date': portfolio_dates[1:], 'Portfolio Value': portfolio_values[1:]})

                # Create a figure with a smaller size
                fig, ax = plt.subplots(figsize=(8, 6))

                # Create a line plot using Seaborn
                sns.lineplot(data=chart_data, x='Date', y='Portfolio Value', ax=ax)
                ax.set_xticklabels(chart_data['Date'], rotation=45)
                ax.set_title('Portfolio Value Over Time')

                # Display the plot
                st.pyplot(fig)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.header('📝 Transaction Logs')
    def view_transactions(user_id):
        # Fetch transaction data from the database
        c.execute('''SELECT * FROM transactions WHERE user_id = ?''', (user_id,))
        transaction_data = c.fetchall()

        if transaction_data:
            # Create a DataFrame with the transaction data
            df_transactions = pd.DataFrame(transaction_data,
                                           columns=['ID', 'User ID', 'Stock ID', 'Transaction Type', 'Shares', 'Price',
                                                    'Date'])

            # Display the transaction table
            st.table(df_transactions)
        else:
            st.info('No transactions found.')
    st.markdown("<hr>", unsafe_allow_html=True)
    # Call the function to display the transaction data
    view_transactions(get_session_state())

    st.markdown("<hr>", unsafe_allow_html=True)
    st.header('💹 Compare portfolio performance against a benchmark')
    st.markdown("<hr>", unsafe_allow_html=True)
    benchmark_data = [('SPY', 'S&P 500', 'Large-cap stocks'),
                      ('EFA', 'MSCI EAFE', 'Developed international stocks'),
                      ('AGG', 'Bloomberg Barclays Aggregate Bond', 'U.S. investment-grade bonds')]
    benchmark_dropdown_options = [f'{row[1]} ({row[0]})' for row in benchmark_data]
    benchmark_dropdown_index = st.selectbox('Select a benchmark', range(len(benchmark_data)),
                                            format_func=lambda i: benchmark_dropdown_options[i])
    benchmark_id = benchmark_dropdown_index + 1
    start_date = st.date_input('Start date')
    end_date = st.date_input('End date')

    if start_date >= end_date:
        st.warning('Start date should be earlier than end date.')
    else:
        if st.button('Calculate performance'):
            portfolio_data = get_user_portfolio(st.session_state.user_id)
            if portfolio_data:
                def calculate_portfolio_value2(portfolio_data, start_date, end_date):
                    portfolio_dates = pd.date_range(start=start_date, end=end_date)
                    portfolio_values = []

                    for date in portfolio_dates:
                        date_str = date.strftime('%Y-%m-%d')
                        total_value = 0.0

                        for row in portfolio_data:
                            purchase_date = datetime.datetime.strptime(row[6], '%Y-%m-%d').date()
                            stock_data = yf.download(row[1], start=purchase_date, end=date_str)
                            if not stock_data.empty:
                                current_price = stock_data['Close'].iloc[-1]
                                stock_value = (current_price - purchase_price) * row[4]
                                total_value += stock_value

                        if date == purchase_date:
                            for row in portfolio_data:
                                stock_val = purchase_price * row[4]
                                total_value = total_value + stock_val

                        portfolio_values.append(total_value)
                        for i in range(0, len(portfolio_values)):
                            data = portfolio_values[i]
                            if data == 0:
                                portfolio_values[i] = portfolio_values[i - 1]
                    return portfolio_dates, portfolio_values
                # Get the earliest purchase date from the portfolio
                earliest_date = min(row[6] for row in portfolio_data)
                # Calculate the portfolio value for each day between the earliest purchase date and current date
                current_date = datetime.date.today()
                portfolio_dates, portfolio_values = calculate_portfolio_value2(portfolio_data, earliest_date,
                                                                               current_date)
                def get_benchmark_prices2(benchmark_ticker, start_date, end_date):
                    benchmark_dates = pd.date_range(start=start_date, end=end_date)
                    benchmark_prices = []
                    previous_price = None
                    for date in benchmark_dates:
                        try:
                            date_str = date.strftime('%Y-%m-%d')
                            ndate=increment_date(date_str);
                            prices = yf.download(benchmark_ticker, start=date_str, end=ndate)['Close']

                            if prices.empty:
                                if previous_price is not None:
                                    benchmark_prices.append(previous_price)
                                else:
                                    raise ValueError(
                                        f"No data available for benchmark ticker '{benchmark_ticker}' on {date_str}")
                            else:
                                price = prices.iloc[0]
                                benchmark_prices.append(price)
                                previous_price = price
                        except:
                            benchmark_prices.append(0)

                    return benchmark_dates, benchmark_prices


                # Get benchmark data
                benchmark_ticker = benchmark_data[benchmark_dropdown_index][0]
                benchmark_prices = get_benchmark_prices2(benchmark_ticker, earliest_date , current_date)[1]

                # Print the lengths of the arrays for debugging
                print("Portfolio dates length:", len(portfolio_dates),portfolio_dates)
                print("Portfolio values length:", len(portfolio_values),portfolio_values)
                print("Benchmark prices length:", len(benchmark_prices),benchmark_prices )

                # Create a DataFrame with the portfolio and benchmark data
                performance_data = pd.DataFrame({
                    'Date': portfolio_dates,
                    'Portfolio Value': portfolio_values,
                    'Benchmark Value': benchmark_prices
                })
                # Calculate portfolio returns and benchmark returns
                portfolio_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
                benchmark_return = (benchmark_prices[-1] - benchmark_prices[0]) / benchmark_prices[0]

                # Create a DataFrame with the portfolio and benchmark data
                performance_data = pd.DataFrame({'Date': portfolio_dates,
                                                 'Portfolio Value': portfolio_values,
                                                 'Benchmark Value': benchmark_prices})

                # Display the performance data
                st.dataframe(performance_data)
                st.write(f'Portfolio value: ${portfolio_values[-1]:.2f}')
                st.write(f'Benchmark value: ${benchmark_prices[-1]:.2f}')
                st.write(f'Portfolio return: {portfolio_return:.2%}')
                st.write(f'Benchmark return: {benchmark_return:.2%}')
                # Calculate daily returns for portfolio and benchmark
                portfolio_daily_returns = performance_data['Portfolio Value'].pct_change()
                benchmark_daily_returns = performance_data['Benchmark Value'].pct_change()

                # Remove last few data points if their values are the same
                while len(portfolio_daily_returns) > 0 and portfolio_daily_returns.iloc[-1] == 0:
                    portfolio_daily_returns = portfolio_daily_returns[:-1]
                    benchmark_daily_returns = benchmark_daily_returns[:-1]

                # Align the lengths of portfolio and benchmark returns
                min_length = min(len(portfolio_daily_returns), len(benchmark_daily_returns))
                portfolio_daily_returns = portfolio_daily_returns[:min_length]
                benchmark_daily_returns = benchmark_daily_returns[:min_length]

                # Create a DataFrame for plotting
                returns_data = pd.DataFrame({
                    'Date': performance_data['Date'][:min_length],
                    'Portfolio Return': portfolio_daily_returns,
                    'Benchmark Return': benchmark_daily_returns
                })

                # Create a line plot using seaborn
                sns.set_style('darkgrid')
                plt.figure(figsize=(12, 6))
                sns.lineplot(x='Date', y='value', hue='variable', data=pd.melt(returns_data, ['Date']))
                plt.xlabel('Date')
                plt.ylabel('Return')
                plt.title('Daily Portfolio and Benchmark Returns')

                # Rotate x-axis labels
                plt.xticks(rotation=45)

                # Display the plot
                st.pyplot(plt)
else:
    st.header("Login to continue")
