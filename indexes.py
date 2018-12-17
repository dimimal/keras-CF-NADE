import time
import os
import shutil
import re
import json

import numpy as np

from pyspark.sql.functions import col, udf

DATA_PATH = os.path.join(os.path.dirname(__file__), 'ui_data')

def create_user_index(ui_mat_rdd, export=False):
    """
    Indexes the users and saves the index in a dictionary. The indexes are
    then saved to disk.
    NEEDS TO BE REVISITED
    Args:
            ui_mat_rdd: The UI matrix in an RDD.
    Returns:
            user_index: A dictionary with the indexes.
    """

    user_index = ui_mat_rdd.map(lambda (usrId,docId,value): usrId) \
            .distinct().zipWithIndex().collect()

    user_index = dict(zip([usr[0] for usr in user_index],[usr[1] for usr in user_index]))

    if export:
        # with open('/local/agidiotis/ui_data/user_index.json', 'w') as fp:
        with open(DATA_PATH, 'user_index.json', 'w') as fp:
            json.dump(user_index, fp)

    return user_index


def create_doc_index(ui_mat_rdd, export=False):
    """
    Indexes the docs and saves the index in a dictionary. The indexes are
    then saved to disk. The indexing allows us to use internal integer IDs
    for the users and documents instead of real IDs. The internal IDs are used
    to index the different model matrices efficiently.
    Args:
            ui_mat_rdd: The UI matrix in an RDD.
    Returns:
            doc_index: A dictionary with the indexes.
    """

    doc_index = ui_mat_rdd.map(lambda (usrId,docId,value): docId) \
            .distinct() \
            .zipWithIndex() \
            .collect()

    doc_index = dict(zip([doc[0] for doc in doc_index],[doc[1] for doc in doc_index]))

    if export:
        with open(DATA_PATH, 'doc_index.json', 'w') as fp:
            json.dump(doc_index, fp)

    return doc_index


def load_indexes():
    """
        Load the user and document indexes used for the model. Here the indexes
        are also reversed in order to be used for the reverse mapping.
        Returns:
                doc_index: A dictionary with the doc -> docID index.
                user_index: A dictionary with the user -> userID index.
                inv_doc_index: A dictionary with the docID -> doc index.
                inv_user_index: A dictionary with the userID -> user index.
    """

    with open(DATA_PATH, 'doc_index.json') as json_file:
        json_string = json_file.read()
        doc_index = json.loads(json_string)
        doc_index = dict([int(k), int(v)] for k,v in doc_index.items())

    with open(DATA_PATH, 'user_index.json') as json_file:
        json_string = json_file.read()
        user_index = json.loads(json_string)
        user_index = dict([int(k), int(v)] for k,v in user_index.items())

    inv_doc_index = {v: k for k, v in doc_index.iteritems()}
    inv_user_index = {v: k for k, v in user_index.iteritems()}

    return doc_index, user_index, inv_doc_index, inv_user_index


def map_recommendations(
        recommendations,
        inv_user_index,
        inv_doc_index,
        mode='usr2doc'):

    """
    Map internal IDs to real IDs. Supports three modes depending on the
    recommendations shape.
    Args:
            recommendations: A list of tuples to be mapped from internal IDs to
                    real IDs. The tuples can have one of the following shapes.
                    (userID,docID,value),(userID,userID,value)or (docID,docID,value).
            inv_user_index: A dictionary with the userID -> user index.
            inv_doc_index: A dictionary with the docID -> doc index.
            mode: One of (usr2doc,usr2usr,doc2doc). This depends on the shape of
                    the recommendations.
    Returns:
            recommendations: A list of tuples mapped from internal IDs to real IDs.
    """

    if mode == 'usr2usr':
        recommendations = [(inv_user_index[rec[0]],inv_user_index[rec[1]],rec[2]) for rec in recommendations]
    elif mode == 'doc2doc':
        recommendations = [(inv_doc_index[rec[0]],inv_doc_index[rec[1]],rec[2]) for rec in recommendations]
    else:
        recommendations = [(inv_user_index[rec[0]],inv_doc_index[rec[1]],rec[2]) for rec in recommendations]

    return recommendations
