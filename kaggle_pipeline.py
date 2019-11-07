import re, os, sys
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, storage, firestore

ID_COLUMN = 'id'
PRICE_COLUMN = 'precio'
CSV_PATH = './data'

CRED_FIRESTORE = './serviceAccountKey.json'

def _firebase_upload(filename, path_to_csv, description):
    cred = credentials.Certificate(CRED_FIRESTORE)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'datos-tp2-20192c.appspot.com'
    })
    # Upload Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob(filename)
    blob.upload_from_filename(path_to_csv)
    print('Subido a Firebase')
    blob.make_public()
    csv_firebase_url = blob.public_url
    # Upload Firebase Firestore
    store = firestore.client()
    doc_ref = store.collection(u'csv')
    doc_ref.add({u'url': csv_firebase_url, u'description': description, u'file_name': filename})

def _df_to_csv(df):
    dt_string = datetime.now().strftime("%d-%m-%Y-%H%M%S")
    output_file = dt_string + '.csv'
    output_dir = Path(CSV_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not df.to_csv(output_dir/output_file) == None:
        return output_dir, output_file
    print('No se guardo el csv')
    return None

def _get_dir_filename(path):
    res = re.findall("(?:[^\]\[\/]+|\[[^\]\[]+\])+", path)
    return '/'.join(res[:-1]), res[len(res)-1]

# def kaggle_upload(df, description):
def kaggle_upload(csv_path, description):
    # competition_columns = [ID_COLUMN, PRICE_COLUMN]
    # df_upload = df[competition_columns]
    # csv_dir, csv_file = _df_to_csv(df_upload)
    # if path == None:
    #     return
    csv_dir, csv_file = _get_dir_filename(csv_path)
    _firebase_upload(csv_file, csv_dir + '/' + csv_file, description)
    os.system('kaggle competitions submit -c Inmuebles24 -f ' + csv_dir + '/' + csv_file + ' -m \"Message\"')
    print('Subido a Kaggle')

if __name__== "__main__":
  csv_path = sys.argv[1]
  description = sys.argv[2]
  kaggle_upload(csv_path, description)
