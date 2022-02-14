#!/usr/bin/env python


import argparse
from typing import Any, Dict
from datetime import datetime
from pprint import pprint
import sqlite3
import time
from kaggle.api.kaggle_api_extended import KaggleApi


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Kaggle submissions viewer')
    parser.add_argument('--name', '-n', required=True, type=str, help='competition name')
    parser.add_argument('--interval', '-i', type=int, default=10, help='fetch interval (sec)')
    return parser.parse_args()


def get_db(dbname: str) -> sqlite3.Connection:
    con = sqlite3.connect(dbname)
    con.row_factory = sqlite3.Row
    return con


def get_kaggle_api() -> KaggleApi:
    api = KaggleApi()
    api.authenticate()
    return api


def regist_submission(con: sqlite3.Connection, sub: Dict[str, Any]) -> None:
    cur = con.cursor()
    sql = (
        'INSERT INTO `submissions` (id, file_name, description, status, score, running_time)'
        'VALUES (?,?,?,?,?,?)'
        'ON CONFLICT(id)'
        'DO UPDATE SET running_time = ?'
    )
    if sub['status'] == 'pending':
        now = datetime.fromisoformat(datetime.isoformat(datetime.utcnow()).split('.')[0])
        raw_date = sub['date'].rstrip('Z').split('.')[0]
        submit_datetime = datetime.fromisoformat(raw_date)
        td = now - submit_datetime
        running_time = round(td.seconds/60/60, 2)

        cur.execute(sql, (
            sub['ref'],
            sub['fileName'],
            sub['description'],
            sub['status'],
            sub['publicScore'],
            running_time,
            running_time
        ))
        con.commit()
    elif sub['status'] == 'complete':
        sql = (
            'UPDATE `submissions`'
            'SET status = "complete", score = ?'
            'WHERE id = ?'
        )
        cur.execute(sql, (sub['publicScore'], int(sub['ref'])))
        con.commit()
    else:
        pass
    
    return    


def check_pending(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    sql = 'SELECT description, running_time FROM submissions WHERE status="pending"'
    cur.execute(sql)
    rows = cur.fetchall()
    print(f'check {len(rows)} pending submissions')
    for row in rows:
        print(f'{row["description"]}, {row["running_time"]} hours')


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    kaggle_api = get_kaggle_api()
    con = get_db('kaggle.db')
    daemon = True
    if daemon:
        while True:
            subs = kaggle_api.competitions_submissions_list(args.name, page=1)
    
            for sub in subs:
                regist_submission(con, sub)
    
            check_pending(con)
            print('-------------')
            time.sleep(60*args.interval)
    
    con.close()

