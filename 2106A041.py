import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, hinge_loss

def load_data():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not filepath:
        return
    global dataset
    dataset = pd.read_csv(filepath)
    data_text.insert(tk.END, dataset.head().to_string() + "\n")

def handle_missing_data():
    global dataset
    if dataset is None:
        messagebox.showerror("Error", "No dataset loaded!")
        return
    
    method = missing_var.get()
    if method == "Mean Imputation":
        imputer = SimpleImputer(strategy="mean")
        dataset.iloc[:, :] = imputer.fit_transform(dataset)
    elif method == "Interpolation":
        dataset = dataset.interpolate()
    elif method == "Forward Fill":
        dataset = dataset.fillna(method="ffill")
    elif method == "Backward Fill":
        dataset = dataset.fillna(method="bfill")
    
    messagebox.showinfo("Info", "Missing values handled using " + method)

def train_model():
    global dataset
    if dataset is None:
        messagebox.showerror("Error", "No dataset loaded!")
        return
    
    target_col = target_entry.get().strip()
    if target_col not in dataset.columns:
        messagebox.showerror("Error", f"Invalid target column! Available columns: {dataset.columns.tolist()}")
        return
    
    X = dataset.drop(columns=[target_col])
    y = dataset[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model_type = model_var.get()
    loss_type = loss_var.get()
    
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Ridge Regression":
        model = Ridge()
    elif model_type == "Lasso Regression":
        model = Lasso()
    elif model_type == "Huber Regression":
        model = HuberRegressor()
    elif model_type == "Logistic Regression":
        model = LogisticRegression()
    elif model_type == "SVM (Regression)":
        model = SVR(kernel=kernel_var.get(), C=float(C_entry.get()), epsilon=float(epsilon_entry.get()))
    elif model_type == "SVM (Classification)":
        model = SVC(kernel=kernel_var.get())
    elif model_type == "Naïve Bayes":
        priors = None if not prior_entry.get() else list(map(float, prior_entry.get().split(',')))
        model = GaussianNB(var_smoothing=float(var_smoothing_entry.get()), priors=priors)
    else:
        messagebox.showerror("Error", "Invalid model selection!")
        return
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if loss_type == "MSE":
        error = mean_squared_error(y_test, y_pred)
    elif loss_type == "MAE":
        error = mean_absolute_error(y_test, y_pred)
    elif loss_type == "Cross-Entropy":
        error = log_loss(y_test, y_pred)
    elif loss_type == "Hinge Loss":
        error = hinge_loss(y_test, y_pred)
    else:
        error = "Unknown loss function"
    
    result_text.insert(tk.END, f"Model: {model_type}\nLoss ({loss_type}): {error}\n\n")
    
    plot_results(y_test, y_pred)

def plot_results(y_test, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Model Predictions vs Actual")
    
    for widget in plot_frame.winfo_children():
        widget.destroy()
    
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

root = tk.Tk()
root.title("Enhanced ML GUI")

target_label = ttk.Label(root, text="Target Column:")
target_label.pack()
target_entry = ttk.Entry(root)
target_entry.pack()

model_var = tk.StringVar(value="Linear Regression")
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=["Linear Regression", "Ridge Regression", "Lasso Regression", "Huber Regression", "Logistic Regression", "SVM (Regression)", "SVM (Classification)", "Naïve Bayes"])
model_dropdown.pack()

kernel_var = tk.StringVar(value="rbf")
kernel_dropdown = ttk.Combobox(root, textvariable=kernel_var, values=["linear", "rbf", "poly"])
kernel_dropdown.pack()

loss_var = tk.StringVar(value="MSE")
loss_dropdown = ttk.Combobox(root, textvariable=loss_var, values=["MSE", "MAE", "Cross-Entropy", "Hinge Loss"])
loss_dropdown.pack()

data_text = tk.Text(root, height=5, width=50)
data_text.pack()

load_button = ttk.Button(root, text="Load Data", command=load_data)
load_button.pack()

missing_var = tk.StringVar(value="Mean Imputation")
missing_dropdown = ttk.Combobox(root, textvariable=missing_var, values=["Mean Imputation", "Interpolation", "Forward Fill", "Backward Fill"])
missing_dropdown.pack()
missing_button = ttk.Button(root, text="Handle Missing Data", command=handle_missing_data)
missing_button.pack()

train_button = ttk.Button(root, text="Train Model", command=train_model)
train_button.pack()

result_text = tk.Text(root, height=5, width=50)
result_text.pack()

plot_frame = ttk.Frame(root)
plot_frame.pack()

root.mainloop()