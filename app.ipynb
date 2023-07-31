{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "95a72b02",
   "metadata": {},
   "source": [
    "# Importing library and data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e8f85963-5468-4138-b981-d3e82c309f84",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import tensorflow_hub as hub\n",
    "import tensorflow_text as text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "dec15dcc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                 0                                                  1\n",
      "0        Household  Paper Plane Design Framed Wall Hanging Motivat...\n",
      "1        Household  SAF 'Floral' Framed Painting (Wood, 30 inch x ...\n",
      "2        Household  SAF 'UV Textured Modern Art Print Framed' Pain...\n",
      "3        Household  SAF Flower Print Framed Painting (Synthetic, 1...\n",
      "4        Household  Incredible Gifts India Wooden Happy Birthday U...\n",
      "...            ...                                                ...\n",
      "50420  Electronics  Strontium MicroSD Class 10 8GB Memory Card (Bl...\n",
      "50421  Electronics  CrossBeats Wave Waterproof Bluetooth Wireless ...\n",
      "50422  Electronics  Karbonn Titanium Wind W4 (White) Karbonn Titan...\n",
      "50423  Electronics  Samsung Guru FM Plus (SM-B110E/D, Black) Colou...\n",
      "50424  Electronics                   Micromax Canvas Win W121 (White)\n",
      "\n",
      "[50425 rows x 2 columns]\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "dt = pd.read_csv('./datasets/ecommerceDataset.csv', header=None)\n",
    "print(dt)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2f8c8a39",
   "metadata": {},
   "source": [
    "## **Data preprocessing**\n",
    "### *Getting all categories*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ab6fa897",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Household' 'Books' 'Clothing & Accessories' 'Electronics']\n"
     ]
    }
   ],
   "source": [
    "categories = dt[0].unique()\n",
    "print(categories)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e5bc0a7c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr:last-of-type th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th colspan=\"4\" halign=\"left\">1</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>unique</th>\n",
       "      <th>top</th>\n",
       "      <th>freq</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Books</th>\n",
       "      <td>11820</td>\n",
       "      <td>6256</td>\n",
       "      <td>Think &amp; Grow Rich About the Author NAPOLEON HI...</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Clothing &amp; Accessories</th>\n",
       "      <td>8670</td>\n",
       "      <td>5674</td>\n",
       "      <td>Diverse Men's Formal Shirt Diverse is a wester...</td>\n",
       "      <td>23</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Electronics</th>\n",
       "      <td>10621</td>\n",
       "      <td>5308</td>\n",
       "      <td>HP 680 Original Ink Advantage Cartridge (Black...</td>\n",
       "      <td>26</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Household</th>\n",
       "      <td>19313</td>\n",
       "      <td>10564</td>\n",
       "      <td>Nilkamal Series-24 Chest of Drawers (Cream Tra...</td>\n",
       "      <td>13</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                            1          \n",
       "                        count unique   \n",
       "0                                      \n",
       "Books                   11820   6256  \\\n",
       "Clothing & Accessories   8670   5674   \n",
       "Electronics             10621   5308   \n",
       "Household               19313  10564   \n",
       "\n",
       "                                                                                \n",
       "                                                                      top freq  \n",
       "0                                                                               \n",
       "Books                   Think & Grow Rich About the Author NAPOLEON HI...   30  \n",
       "Clothing & Accessories  Diverse Men's Formal Shirt Diverse is a wester...   23  \n",
       "Electronics             HP 680 Original Ink Advantage Cartridge (Black...   26  \n",
       "Household               Nilkamal Series-24 Chest of Drawers (Cream Tra...   13  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dt.groupby(0).describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ce58d0c6",
   "metadata": {},
   "source": [
    "### *Handling dataframes*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "11c1c4d2",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(8671, 2)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_accessories = dt[dt[0] == 'Clothing & Accessories']\n",
    "df_accessories.head(10)\n",
    "df_accessories.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8dde9a98",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(11820, 2)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_books = dt[dt[0] == 'Books']\n",
    "df_books.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "f733c46d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(19313, 2)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_household = dt[dt[0] == 'Household']\n",
    "df_household.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d98038cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10621, 2)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_elec = dt[dt[0] == 'Electronics']\n",
    "df_elec.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "31e93b4c",
   "metadata": {},
   "source": [
    "### ***Balancing dataframes***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "7840e9f8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(8671, 2)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_books_downsample = df_books.head(df_accessories.shape[0])\n",
    "df_books_downsample.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4876eb86",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(8671, 2)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_household_downsample = df_household.head(df_accessories.shape[0])\n",
    "df_household_downsample.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7b8f58fe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>39804</th>\n",
       "      <td>Electronics</td>\n",
       "      <td>Dell 19.5V-3.34AMP 65W Laptop Adapter (Without...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39805</th>\n",
       "      <td>Electronics</td>\n",
       "      <td>Bluetooth Dongle USB CSR 4.0 Adapter Receiver ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39806</th>\n",
       "      <td>Electronics</td>\n",
       "      <td>Wi-Fi Receiver 300Mbps, 2.4GHz, 802.11b/g/n US...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39807</th>\n",
       "      <td>Electronics</td>\n",
       "      <td>SanDisk 64GB Class 10 microSDXC Memory Card wi...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39808</th>\n",
       "      <td>Electronics</td>\n",
       "      <td>Gizga Essentials Laptop Power Cable Cord- 3 Pi...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 0                                                  1\n",
       "39804  Electronics  Dell 19.5V-3.34AMP 65W Laptop Adapter (Without...\n",
       "39805  Electronics  Bluetooth Dongle USB CSR 4.0 Adapter Receiver ...\n",
       "39806  Electronics  Wi-Fi Receiver 300Mbps, 2.4GHz, 802.11b/g/n US...\n",
       "39807  Electronics  SanDisk 64GB Class 10 microSDXC Memory Card wi...\n",
       "39808  Electronics  Gizga Essentials Laptop Power Cable Cord- 3 Pi..."
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_elec_downsample = df_elec.head(df_accessories.shape[0])\n",
    "df_elec_downsample.shape\n",
    "df_elec_downsample.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cbf335aa",
   "metadata": {},
   "source": [
    "### ***Merging dataframes***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "b62112df",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>31133</th>\n",
       "      <td>Clothing &amp; Accessories</td>\n",
       "      <td>Woopower 36M Pink for 024M Baby Trouser Top Se...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>31134</th>\n",
       "      <td>Clothing &amp; Accessories</td>\n",
       "      <td>Amour Butterfly Design Sunglasses For Girls 6+...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>31135</th>\n",
       "      <td>Clothing &amp; Accessories</td>\n",
       "      <td>Vaenait Baby 024M Baby Girls Rashguard Swimwea...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>31136</th>\n",
       "      <td>Clothing &amp; Accessories</td>\n",
       "      <td>Amour Butterfly Design Sunglasses For Girls 6+...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>31137</th>\n",
       "      <td>Clothing &amp; Accessories</td>\n",
       "      <td>Kuchipoo Girl's Cotton Regular Fit T-Shirt - P...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48470</th>\n",
       "      <td>Electronics</td>\n",
       "      <td>LG GH24NSD1 Internal SATA DVD Writer The M-DIS...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48471</th>\n",
       "      <td>Electronics</td>\n",
       "      <td>LG GP65NB60 External DVD Writer (Black) LG GP6...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48472</th>\n",
       "      <td>Electronics</td>\n",
       "      <td>PIONEER DVD PLAYER DV-3052V Pioneer DV-3052 Mu...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48473</th>\n",
       "      <td>Electronics</td>\n",
       "      <td>LG DP546 DVD Player (Black) DivX-This is a for...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48474</th>\n",
       "      <td>Electronics</td>\n",
       "      <td>PODOFO 10.9cm Foldable TFT Color LCD Car Monit...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>34684 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                            0   \n",
       "31133  Clothing & Accessories  \\\n",
       "31134  Clothing & Accessories   \n",
       "31135  Clothing & Accessories   \n",
       "31136  Clothing & Accessories   \n",
       "31137  Clothing & Accessories   \n",
       "...                       ...   \n",
       "48470             Electronics   \n",
       "48471             Electronics   \n",
       "48472             Electronics   \n",
       "48473             Electronics   \n",
       "48474             Electronics   \n",
       "\n",
       "                                                       1  \n",
       "31133  Woopower 36M Pink for 024M Baby Trouser Top Se...  \n",
       "31134  Amour Butterfly Design Sunglasses For Girls 6+...  \n",
       "31135  Vaenait Baby 024M Baby Girls Rashguard Swimwea...  \n",
       "31136  Amour Butterfly Design Sunglasses For Girls 6+...  \n",
       "31137  Kuchipoo Girl's Cotton Regular Fit T-Shirt - P...  \n",
       "...                                                  ...  \n",
       "48470  LG GH24NSD1 Internal SATA DVD Writer The M-DIS...  \n",
       "48471  LG GP65NB60 External DVD Writer (Black) LG GP6...  \n",
       "48472  PIONEER DVD PLAYER DV-3052V Pioneer DV-3052 Mu...  \n",
       "48473  LG DP546 DVD Player (Black) DivX-This is a for...  \n",
       "48474  PODOFO 10.9cm Foldable TFT Color LCD Car Monit...  \n",
       "\n",
       "[34684 rows x 2 columns]"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dfs_array = [df_accessories, df_books_downsample, df_household_downsample, df_elec_downsample]\n",
    "df_merged = pd.concat(dfs_array)\n",
    "df_merged"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "28d982ff",
   "metadata": {},
   "source": [
    "### Encoding texts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7e8cdba6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 1, 0, 0],\n",
       "       [0, 1, 0, 0],\n",
       "       [0, 1, 0, 0],\n",
       "       ...,\n",
       "       [0, 0, 1, 0],\n",
       "       [0, 0, 1, 0],\n",
       "       [0, 0, 1, 0]])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from keras.utils import np_utils\n",
    "import numpy as np\n",
    "\n",
    "labels = df_merged[0]\n",
    "texts = df_merged[1]\n",
    "# encode class values as integers\n",
    "encoder = LabelEncoder()\n",
    "encoder.fit(labels)\n",
    "encoded_Y = encoder.transform(labels)\n",
    "# convert integers to dummy variables (i.e. one hot encoded)\n",
    "y_cat = np_utils.to_categorical(encoded_Y)\n",
    "y_cat = y_cat.astype(int)\n",
    "y_cat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "33f56db4",
   "metadata": {},
   "outputs": [],
   "source": [
    "cols = df_merged.select_dtypes(include=['object'])\n",
    "for col in cols.columns.values:\n",
    "    df_merged[col] = df_merged[col].fillna('')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4c3d9e31",
   "metadata": {},
   "source": [
    "### ***Splitting dataframe***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "58c9b901",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sklearn\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(df_merged[1], y_cat, train_size=0.2)\n",
    "#y_train = y_train.astype('string')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bef83d57",
   "metadata": {},
   "source": [
    "### ***Importing BERT and getting embeding vectors for data***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "9bca5f0a",
   "metadata": {},
   "outputs": [],
   "source": [
    "bert_preprocess = hub.KerasLayer(\"https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2\")\n",
    "bert_encoder = hub.KerasLayer(\"https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e171c03c",
   "metadata": {},
   "source": [
    "**Example getting embeding of sentence**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "1e474ace",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(2, 768), dtype=float32, numpy=\n",
       "array([[-0.84351695, -0.5132727 , -0.88845736, ..., -0.74748874,\n",
       "        -0.75314736,  0.91964495],\n",
       "       [-0.87208354, -0.50543964, -0.94446677, ..., -0.8584749 ,\n",
       "        -0.7174534 ,  0.88082975]], dtype=float32)>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def get_sentence_embeding(sentences):\n",
    "    preprocessed_text = bert_preprocess(sentences)\n",
    "    return bert_encoder(preprocessed_text)['pooled_output']\n",
    "\n",
    "get_sentence_embeding([\n",
    "    \"500$ discount. hurry up\", \n",
    "    \"Bhavin, are you up for a volleybal game tomorrow?\"]\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac747a7a",
   "metadata": {},
   "source": [
    "**Building model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "c450c2d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import Sequential\n",
    "from keras.layers import Dense\n",
    "from keras.layers import Dropout\n",
    "from keras.optimizers import Adam\n",
    "\n",
    "\n",
    "#BERT Layer\n",
    "text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name = \"text\")\n",
    "preprocessed_inputs = bert_preprocess(text_input)\n",
    "encoded_outputs = bert_encoder(preprocessed_inputs)\n",
    "\n",
    "#Neural network\n",
    "layer = tf.keras.layers.Dropout(0.2, name=\"dropout\")(encoded_outputs['pooled_output'])\n",
    "layer= tf.keras.layers.Dense(4, activation='softmax', name=\"output\")(layer)\n",
    "\n",
    "#Construct the final model\n",
    "model = tf.keras.Model(inputs=[text_input], outputs=[layer])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "bdf90b7f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"model\"\n",
      "__________________________________________________________________________________________________\n",
      " Layer (type)                   Output Shape         Param #     Connected to                     \n",
      "==================================================================================================\n",
      " text (InputLayer)              [(None,)]            0           []                               \n",
      "                                                                                                  \n",
      " keras_layer (KerasLayer)       {'input_word_ids':   0           ['text[0][0]']                   \n",
      "                                (None, 128),                                                      \n",
      "                                 'input_type_ids':                                                \n",
      "                                (None, 128),                                                      \n",
      "                                 'input_mask': (Non                                               \n",
      "                                e, 128)}                                                          \n",
      "                                                                                                  \n",
      " keras_layer_1 (KerasLayer)     {'pooled_output': (  109482241   ['keras_layer[0][0]',            \n",
      "                                None, 768),                       'keras_layer[0][1]',            \n",
      "                                 'default': (None,                'keras_layer[0][2]']            \n",
      "                                768),                                                             \n",
      "                                 'encoder_outputs':                                               \n",
      "                                 [(None, 128, 768),                                               \n",
      "                                 (None, 128, 768),                                                \n",
      "                                 (None, 128, 768),                                                \n",
      "                                 (None, 128, 768),                                                \n",
      "                                 (None, 128, 768),                                                \n",
      "                                 (None, 128, 768),                                                \n",
      "                                 (None, 128, 768),                                                \n",
      "                                 (None, 128, 768),                                                \n",
      "                                 (None, 128, 768),                                                \n",
      "                                 (None, 128, 768),                                                \n",
      "                                 (None, 128, 768),                                                \n",
      "                                 (None, 128, 768)],                                               \n",
      "                                 'sequence_output':                                               \n",
      "                                 (None, 128, 768)}                                                \n",
      "                                                                                                  \n",
      " dropout (Dropout)              (None, 768)          0           ['keras_layer_1[0][13]']         \n",
      "                                                                                                  \n",
      " output (Dense)                 (None, 4)            3076        ['dropout[0][0]']                \n",
      "                                                                                                  \n",
      "==================================================================================================\n",
      "Total params: 109,485,317\n",
      "Trainable params: 3,076\n",
      "Non-trainable params: 109,482,241\n",
      "__________________________________________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "a404687c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.optimizers import Adam\n",
    "\n",
    "model.compile(Adam(learning_rate = 0.007), \"categorical_crossentropy\", metrics=[tf.keras.metrics.BinaryAccuracy(name = 'Accuracy'),\n",
    "                       tf.keras.metrics.Precision(name = 'Precision'), \n",
    "                       tf.keras.metrics.Recall(name = 'Recall')])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "2fde521e",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "217/217 [==============================] - 999s 5s/step - loss: 0.5563 - Accuracy: 0.9037 - Precision: 0.8312 - Recall: 0.7712\n",
      "Epoch 2/5\n",
      "217/217 [==============================] - 991s 5s/step - loss: 0.5571 - Accuracy: 0.9046 - Precision: 0.8299 - Recall: 0.7778\n",
      "Epoch 3/5\n",
      "217/217 [==============================] - 988s 5s/step - loss: 0.5303 - Accuracy: 0.9123 - Precision: 0.8468 - Recall: 0.7928\n",
      "Epoch 4/5\n",
      "217/217 [==============================] - 950s 4s/step - loss: 0.5452 - Accuracy: 0.9107 - Precision: 0.8402 - Recall: 0.7938\n",
      "Epoch 5/5\n",
      "217/217 [==============================] - 967s 4s/step - loss: 0.5547 - Accuracy: 0.9106 - Precision: 0.8387 - Recall: 0.7956\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x2746f9b57b0>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x_train, y_train, epochs = 5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ab2a42f8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
