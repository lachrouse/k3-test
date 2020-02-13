import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
import numpy as np
pd.plotting.register_matplotlib_converters()


def month():
    # Returns datetime type of previous month and the month previous to that. Used for naming the file outputs in
    # data updater function and kthree function
    today = dt.today()

    if today.month == 1:
        year = dt.now().year - 1
        current_month = today.replace(month=12, year=year)
        previous_month = today.replace(month=11, year=year)

    elif today.month == 2:
        year = dt.now().year - 1
        current_month = today.replace(month=1)
        previous_month = today.replace(month=12, year=year)

    else:
        current = dt.now().month - 1 or 12
        current_month = today.replace(month=current)
        previous = dt.now().month - 2 or 12
        previous_month = today.replace(month=previous)

    current_month_str = current_month.strftime('%Y-%m')
    previous_month_str = previous_month.strftime('%Y-%m')

    return current_month_str, previous_month_str


def data_updater(sales_data_doc):
    # function appends the current months sales data to the previous months master sheet. Also return the name of
    # the current months file so it can be called in the kthree function

    current_month, previous_month = month()
    df = pd.read_excel(sales_data_doc)
    output_df = pd.read_excel("k3 data output %s.xlsx" % previous_month.strftime('%Y%m'))
    output_df = output_df.append(df).reset_index()
    output_df.to_excel("k3 data output %s.xlsx" % current_month.strftime('%Y%m'))

    return "k3 data output %s.xlsx" % current_month.strftime('%Y%m')


def kthree(sales_data_doc, salesperson_doc):
    # Calculates the kthree status based on the updated sales data and outputs the calculated information to excel.

    # reads excel file and sorts the values by date.
    df = pd.read_excel(sales_data_doc).sort_values(['Date'])

    # Formats date to YYYY-MM
    df['Date'] = df['Date'].apply(lambda x: dt.strftime(x, '%Y-%m'))

    # reads excel file and creates dataframe of salespeople and associated accounts
    person = pd.read_excel(salesperson_doc)

    # Creates a set of unique accounts
    unique_set = np.sort(df["Account No"].unique())

    # Creates groupby object of accounts.
    grouped = df.groupby('Account No')

    # Intialises empty dataframe
    kthree_df = pd.DataFrame()

    # Iterates through unique accounts list and creates a dataframe for each account. Also initilaises dataframe
    # columns and cp counter
    for i in range(0, len(pd.Index(unique_set))):
        acc = grouped.get_group(unique_set[i]).reset_index().drop(columns="index")
        acc['mean'] = 0
        acc['cp'] = 0
        cp = 0

        # Iterates through each row in the account specific dataframe and evaluates K3 Status
        for ii in range(0, len(pd.Index(acc))):

            # if first step of iteration it will set mean profit equal to profit
            if ii == 0:
                acc.iloc[[ii], 3] = acc.iloc[ii]['Profit']

                # update counter according to profit
                if acc.iloc[ii]['Profit'] == 0:
                    cp = 0
                else:
                    cp += 1
                acc.iloc[[ii], 4] = cp

            # sets second months mean profit equal to the average profit of the first two months
            elif ii == 1:
                acc.iloc[[ii], 3] = acc.iloc[ii - 1:ii + 1]['Profit'].mean()

                # Updates counter according to profit
                if acc.iloc[ii - 1:ii + 1]['Profit'].mean() == 0:
                    cp = 0
                else:
                    cp += 1
                acc.iloc[[ii], 4] = cp

            # calculates mean profit across the last x number of months. the number of months to look back at is
            # calculated using the counter
            else:
                acc.iloc[[ii], 3] = acc.iloc[ii - cp:ii + 1]['Profit'].mean()

                # Checks the average profit of the previous three entries. If the average profit == 0 the counter is
                # reset to 0.
                if acc.iloc[ii - 2:ii + 1]['Profit'].mean() == 0:
                    cp = 0
                else:
                    cp += 1
                acc.iloc[[ii], 4] = cp

                if cp == 0:
                    acc.iloc[[ii], 3] = 0

        # Creates new columns that annualises the mean profit
        acc['annualised profit'] = acc['mean'] * 12

        # Assigns K3 status based on annulaised profit
        acc['K3 status'] = acc['annualised profit'].apply(lambda x: 'K3' if x > 3000 else ('non-K3' if x > 0 else 'non-trading'))

        # adds the newly created acc dataframe new kthree dataframe
        kthree_df = kthree_df.append(acc, ignore_index=True)

    # Merges Salesperson data with Account data
    kthree_df = kthree_df.merge(person, on='Account No', how='left')

    pd.set_option('display.max_columns', None, 'display.width', None, 'display.max_rows', None)

    return kthree_df


def plot_kthree(team, rep, sales_data_doc, salesperson_doc):

    # creates pivot tables summarising kthree data set
    no_k3_pivot = pd.pivot_table(kthree(sales_data_doc, salesperson_doc),
                                 values='cp',
                                 index='Date',
                                 columns=['Team', 'Salesperson', 'K3 status'],
                                 aggfunc='count')

    ave_profit_pivot = pd.pivot_table(kthree(sales_data_doc, salesperson_doc),
                                      values='annualised profit',
                                      index='Date',
                                      columns=['Team', 'Salesperson', 'K3 status'],
                                      aggfunc='mean')

    # creates data frames that are slices of the no_k3_pivot and ave_profit_pivot based on the salesperson
    count_section = no_k3_pivot.iloc[:, no_k3_pivot.columns.get_level_values(1) == rep].tail(12)
    count_total = no_k3_pivot.iloc[:, no_k3_pivot.columns.get_level_values(1) == rep].tail(12)
    ap_section = ave_profit_pivot.iloc[:, ave_profit_pivot.columns.get_level_values(1) == rep].tail(12)
    ap_total = ave_profit_pivot.iloc[:, ave_profit_pivot.columns.get_level_values(1) == rep].tail(12)

    count_total[(team, rep, 'combined')] = count_total[(team, rep, 'K3')] + count_total[(team, rep, 'non-K3')]
    ap_total[(team, rep, 'combined')] = ap_total[(team, rep, 'K3')] + ap_total[(team, rep, 'non-K3')]

    # initiate plot figure
    fig = plt.figure()
    fig.set_size_inches(15, 10)
    fig.suptitle(rep, fontsize=20)

    # creates a list of the data frames that each graph is based upon
    ax_data = [count_section,
               ap_section,
               count_total[(team, rep, 'combined')],
               ap_total[(team, rep, 'combined')]]

    print('ax_data', ax_data)

    # iterates through the items in ax_data and creates a plot for each item.
    for i, data in enumerate(ax_data):
        i += 1
        plt.subplot(2, 2, i)
        plt.plot_date(data.index,
                      data,
                      linestyle='-')

        if i < 3:
            for ii, a in enumerate(data):
                for iii in range(0, len(data.index)):
                    x = data.index[iii]
                    y = data[(team, rep, a[2])][iii]
                    plt.annotate('%.0f' % y,
                                 xy=(x, y + data[(team, rep, 'K3')].max()*0.025),
                                 xycoords='data',
                                 rotation=45)
            plt.ylim([data[(team, rep, 'non-trading')].min() - 1, data[(team, rep, 'K3')].max()*1.06])
            plt.xticks(data.index, rotation=45)

        else:
            for iv, z in enumerate(data):
                x = data.index[iv]
                y = data[iv]
                plt.annotate('%.0f' % y,
                             xy=(x, y*1.01),
                             xycoords='data',
                             rotation=45)
            plt.ylim([data.min()*.90, data.max()*1.1])
            plt.xticks(data.index, rotation=45)

    plt.show()
    return fig


def create_table(team, rep, sales_data_doc, salesperson_doc):

    df = kthree(sales_data_doc, salesperson_doc)
    df_isdate = df[df['Date'] == '2019-02-28']
    df_rep = df_isdate[df['Salesperson'] == rep]

    highest_gp = pd.pivot_table(df_rep,
                                values='annualised profit',
                                index='Account No',
                                columns=['Team', 'Salesperson'],
                                aggfunc='mean')

    sorted_df = highest_gp.sort_values(by=[('BWC', 'Lachlan')], ascending=False)
    print(sorted_df)

    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    row_text = [x for x in sorted_df.index.array]
    ax.table(cellText=sorted_df.head(5).values,
             colLabels=sorted_df.columns,
             rowLabels=row_text[:5],
             loc='center')
    fig.tight_layout()
    plt.show()


def plot_totals(sales_data_doc, salesperson_doc):
    # Plots the business totals graphs

    k3_by_team = pd.pivot_table(kthree(sales_data_doc, salesperson_doc),
                                values='cp',
                                index='Date',
                                columns=['Team'],
                                aggfunc='count')

    k3_total = pd.pivot_table(kthree(sales_data_doc, salesperson_doc),
                              values='cp',
                              index='Date',
                              columns=['K3 status'],
                              aggfunc='count')

    profit_by_team = pd.pivot_table(kthree(sales_data_doc, salesperson_doc),
                                    values='annualised profit',
                                    index='Date',
                                    columns=['Team'],
                                    aggfunc='mean')

    profit_total = pd.pivot_table(kthree(sales_data_doc, salesperson_doc),
                                  values='annualised profit',
                                  index='Date',
                                  columns=[],
                                  aggfunc='mean')

    ax_data = [k3_by_team.tail(12),
               profit_by_team.tail(12),
               k3_total.tail(12),
               profit_total.tail(12)]

    # initiate plot figure
    fig = plt.figure()
    fig.set_size_inches(15, 10)
    # fig.suptitle(rep, fontsize=20)

    for i, data in enumerate(ax_data):
        i += 1
        plt.subplot(2, 2, i)
        plt.plot_date(data.index,
                      data,
                      linestyle='-')
        leg_label = []

        # if i < 3:
        for a in data:
            # Data variable has two columns. a iterates through each column.
            leg_label.append(a)

            for ii in range(0, len(data.index)):
                # Iterates through each row in the data frame data

                x = data.index[ii]
                y = data[a][ii]
                plt.annotate('%.0f' % y,
                             xy=(x, y + data[a].max() * 0.025),
                             xycoords='data',
                             rotation=45)
        plt.ylim([data.values.min() * .90, data.values.max() * 1.06])
        plt.xticks(data.index, rotation=45)
        plt.legend(leg_label)

    plt.show()
    return fig


def new_accounts(sales_data_doc, salesperson_doc):
    # returns a table showing new account openings and forecasted annual profit

    data = kthree(sales_data_doc, salesperson_doc)
    cur_date, prev_date = month()
    cur_month = data[data['Date'] == cur_date]
    prev_month = data[data['Date'] != cur_date]
    print('\n', cur_month.head(), '\n', prev_month.head())

    cur_month_list = cur_month['Account No'].to_list()
    prev_month_list = prev_month['Account No'].to_list()
    new = [x for x in cur_month_list if x not in prev_month_list]

    print(cur_month_list, '\n', prev_month_list, '\n', new)

    # returns dataframe showing accounts that are new in the current month
    new_account_df = cur_month[cur_month['Account No'].isin(new)]
    print(new_account_df)

    print(data[data['Account No'] == '98dcba98'])

    return

# kthree("Sales data test 2.0.xlsx", 'Rep and team.xlsx')
# create_table('BWC', 'Lachlan', "Sales data test 2.0.xlsx", 'Rep and team.xlsx')
# plot_kthree('BWC', 'Lachlan', "Sales data test 2.0.xlsx", 'Rep and team.xlsx')
# plot_totals("Sales data test 2.0.xlsx", 'Rep and team.xlsx')
new_accounts("Sales data test 2.0.xlsx", 'Rep and team.xlsx')
