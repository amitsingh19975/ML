# Machine Learning Library in C++

I'm just learning ML and this is a immature library.  I'm slowly building it and learning about model. This lib uses **Eigen** for linear algebra and no other external lib. I created **Frame** similar to **Python Pandas**, **PrettyPrinter** for printing Frame, vector, Eigen, **CSVParser** for parsing csv data, etc.


# Libraries

- **Frame** for data manipulation
- **PrettyPrinter** for printing frame data, vector, Eigen
- **CSVParser** for parsing csv data and directly convert it to Frame
- **PCA** for dimensionality reduction
- **Series**
- **Metric** for confusion matrix
- **Core** base class of model
-	## ML
	* **PolyRegression**
	* **LinearRegression**
	* **LogisticRegression**
	* **KMean** working on it!

# Functions and Members

## Core

- ``virtual void train(Frame* X_train, Frame* y_train)``
-  ``virtual void train(std::shared_ptr<Frame>& X_train, std::shared_ptr<Frame>& y_train)``
- ``virtual Frame* predict(Frame* X_test)``
-  ``Frame* predict(std::shared_ptr<Frame>& X_test)``
## Frame
	
- ``Frame(size_t row, size_t col)``

- ``Frame(std::vector<std::string>& h)``

- ``Frame(std::vector<SeriesUnique> data)``

- ``Frame(SeriesUnique data, std::string header = "")``

- ``Frame(std::vector<SeriesUnique> data,std::vector<std::string> _header)``

- ``template<typename T>
Frame(std::vector<std::vector<T>>& data);``

- ``template<typename T>
Frame(std::vector<T>& data, std::string header = "");``

- ``template<typename T>
Frame(Eigen::MatrixBase<T>& data, std::vector<std::string> headers);``

- ``template<typename T>
Frame(Eigen::MatrixBase<T>& data,std::unordered_map<std::string,int>& label, std::vector<std::string> headers)``

- ``std::vector<std::string> headers):Frame(data,headers)``

- ``template<typename T>
Frame(std::initializer_list<std::initializer_list<T>> data);``

- ``void print(int size = -1, int ind = 5) const noexcept;``

- ``void printHeader(int ind = 5) const noexcept;``

- ``void info() const noexcept;``

- ``bool addSeries(SeriesUnique series);``

- ``bool addSeries(Series* series);``

- ``void labelToNumber(int idx);``

- ``void numberToLabel(int idx);``

- ``void apply(std::function<double(double)> func);``

- ``void apply(std::function<double(double,size_t i)> func);``

- ``void apply(std::function<double(double,size_t i, size_t j)> func);``

- ``Series* at(int i);``

- ``template<typename T> bool addVec(Vec<T>* vec);``

- ``Series* operator[](int i)``

- ``FrameShared operator[](std::initializer_list<std::string> l)``

- ``Series* operator[](std::string_view header)``

- ``size_t rowSize() const noexcept``

- ``size_t colSize() const noexcept``

- ``SeriesUnique dropCol(int idx);``

- ``FrameShared colSlice(size_t start, size_t end = std::numeric_limits<size_t>::max());``

- ``FrameShared colSlice(std::vector<int> l);``

- ``FrameShared colSlice(std::initializer_list<std::string> l);``

- ``Pair split(float trainPercentage = 30, int randomSeed = 42);``

- ``Tuple split(FrameShared& X, FrameShared& y, float trainPercentage = 30, int randomSeed = 42);``

- ``static void randomize(Frame* frame, int seed = 42, int iteration = 50);``

- ``double mean(int col) noexcept``

- ``double sum(int col) noexcept``

- ``double std(int col) noexcept``

- ``double variance(int col) noexcept``

- ``double median(int col) noexcept``

- ``Eigen::VectorXd mean() noexcept;``

- ``Eigen::VectorXd std() noexcept;``

- ``Eigen::VectorXd variance() noexcept;``

- ``Eigen::VectorXd median() noexcept;``

- ``void normalize(double val = 0);``

- ``void Zscore();``

- ``double corrcoef(int i, int j) noexcept;``

- ``Eigen::MatrixXd corrcoefMatrix() noexcept;``

- ``Eigen::MatrixXd cov() noexcept;``

- ``template<typename T = double>
std::unique_ptr<std::unordered_map<T, int>> unique(int col);``

- ``template<typename U = double>
static Vec<U>* cast(Series* data)``

- ``std::unordered_map<std::string,int> getLabel(int idx) noexcept``

- ``void setLabel(std::unordered_map<std::string,int> label, int idx) noexcept``
	
## Series

- ``virtual double mean() const noexcept``
- ``virtual double variance() const noexcept``
- ``virtual double varianceS() const noexcept``
- ``virtual double std() const noexcept``
- ``virtual double stdS() const noexcept``
- ``virtual double median() noexcept``
- ``virtual double max() const noexcept``
- ``virtual double min() const noexcept``
- ``virtual double sum() const noexcept``
- ``virtual std::unique_ptr<std::unordered_map<std::string, int>> unique()``
- ``virtual void apply(std::function<double(double)> func)``
- ``virtual void apply(std::function<double(double,size_t)> func,size_t i)``
- ``virtual void apply(std::function<double(double,size_t,size_t)> func,size_t i)``
- ``virtual void print(int numberOfData = -1) const``
- ``virtual void swap(size_t i, size_t j)``
- ``virtual void push_d(double val)``
- ``virtual void push_s(std::string val)``
- ``virtual double at(int i) const noexcept``
- ``virtual std::string atS(int i) const noexcept``

## Pretty Printer

``static Color ColorHeaderFG``

``static Color ColorHeaderBG``

``static Color ColorBody``

``static  auto  print(Frame* frame,  uint32_t indent =  5,  uint32_t  const maxLine =  2,  uint32_t maxWidth =  20)  ->  void``

``static  auto  print(Frame& frame,  uint32_t indent =  5,  uint32_t  const maxLine =  2,  uint32_t maxWidth =  20)  ->  void``

``static  auto  print(FrameShared& frame,  uint32_t indent =  5,  uint32_t  const maxLine =  2,  uint32_t maxWidth =  20)  ->  void``

``static  auto  print(Series* series,  uint32_t indent =  5,  uint32_t  const maxLine =  2,  uint32_t maxWidth =  20)  ->  void``

``template<typename T> static  auto  print(Vmatrix<T>&  ,  uint32_t indent =  5,  uint32_t  const maxLine =  2,  uint32_t maxWidth =  20)  ->  void``

``template<typename T> static  auto  print(VecArray<T>&  ,  uint32_t indent =  5,  uint32_t  const maxLine =  2,  uint32_t maxWidth =  20)  ->  void``

``template<typename T> static  auto  print(Eigen::MatrixBase<T>&  ,  uint32_t indent =  5,  uint32_t  const maxLine =  2,  uint32_t maxWidth =  20)  ->  void``


## PCA

## CSVParser

## PolyRegression

## LinearRegression

## LogisticRegression

## KMean

