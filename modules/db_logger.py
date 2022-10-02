import sqlite3
import os

db_con = None
cur = None

def initDbConnection(db_name = 'db.sqlite3'):
    exists = os.path.exists('db.sqlite3')
    global db_con, cur

    db_con = sqlite3.connect(db_name, check_same_thread=False)
    cur = db_con.cursor()
    print('connected to database')

    if not exists:
        cur.execute("CREATE TABLE queries(type, fileName, prompt, negative_prompt, prompt_style, prompt_style2, steps, sampler_index, restore_faces, tiling, n_iter, batch_size, cfg_scale, seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras, height, width, enable_hr, scale_latent, denoising_strength)")

def addQuery(type, fileName, prompt, negative_prompt, prompt_style, prompt_style2, steps, sampler_index, restore_faces, tiling, n_iter, batch_size, cfg_scale, seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras, height, width, enable_hr, scale_latent, denoising_strength):
    cur.execute("INSERT INTO queries VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (type, fileName, prompt, negative_prompt, prompt_style, prompt_style2, steps, sampler_index, restore_faces, tiling, n_iter, batch_size, cfg_scale, seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras, height, width, enable_hr, scale_latent, denoising_strength))
    db_con.commit()

def getQueries():
    cur.execute("SELECT * FROM queries")
    return cur.fetchall()