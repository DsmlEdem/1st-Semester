#!/usr/bin/env python3

from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt

def track(filename, start, start_transmit, end_transmit):
    # read the data
    df = pd.read_csv(filename, parse_dates=["When"])
    df.sort_values(by="When", inplace=True)
    # count people
    people = len(set(df['Phone1']) | set(df['Phone2']))
    print("people =", people)
    # find dates
    min_dt = df['When'].min().date()
    max_dt = df['When'].max().date()
    dates = [dt.date() for dt in pd.date_range(min_dt, max_dt)]
    D = len(dates)
    # keep who's sick and who's contageous
    sick = {start: min_dt}
    def contageous(phone, d):
        return phone in sick \
           and start_transmit <= (d - sick[phone]).days < end_transmit
    # simulation
    M = len(df)
    i = 0
    contacts = []
    cases = []
    infected = []
    for d in dates:
        j = i
        sick_before = len(sick)
        while i < M and df.iloc[i, 0].date() == d:
            phone1 = df.iloc[i, 1]
            phone2 = df.iloc[i, 2]
            if contageous(phone1, d) and phone2 not in sick: sick[phone2] = d
            if contageous(phone2, d) and phone1 not in sick: sick[phone1] = d 
            i += 1
        contacts.append(i - j)
        sick_after = len(sick)
        cases.append(sick_after - sick_before)
        infected.append(sick_after)
    # print the rest
    print("max contacts per day =", max(contacts))
    print("min contacts per day =", min(contacts))
    print("average contacts per day =", M/D)
    print("total infected =", len(sick))
    print("max cases per day =", max(cases))
    # plot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(8, 8)
    ax1.bar(dates, cases, color="royalblue")
    ax1.set_title('Cases')
    ax2.bar(dates, infected, color="indianred")
    ax2.set_title('Infected')
    plt.xticks(rotation=90)
    plt.savefig("diagrams.png")

if __name__ == "__main__":
    import sys

    filename = sys.argv[1]
    phone = int(sys.argv[2])
    start_transmit = int(sys.argv[3])
    end_transmit = int(sys.argv[4])

    track(filename, phone, start_transmit, end_transmit)