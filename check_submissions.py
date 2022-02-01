#!/usr/bin/env python

from typing import Any, Dict, List
from kaggle.api.kaggle_api_extended import KaggleApi
from datetime import datetime
from pprint import pprint
import sqlite3


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
    elif sub['status'] == 'complete': ## check id
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


def check_pending(con: sqlite3.Connection):
    cur = con.cursor()
    sql = 'SELECT description, running_time FROM submissions WHERE status="pending"'
    cur.execute(sql)
    rows = cur.fetchall()
    print(f'check {len(rows)} pending submissions')
    for row in rows:
        print(f'{row["description"]}, {row["running_time"]} hours')


if __name__ == '__main__':
    kaggle_api = get_kaggle_api()
    con = get_db('kaggle.db')
    comp_name = 'tensorflow-great-barrier-reef'
    subs = kaggle_api.competitions_submissions_list(comp_name, page=1)
    #pprint(subs)
    for sub in subs:
        regist_submission(con, sub)
    
    check_pending(con)

    con.close()

