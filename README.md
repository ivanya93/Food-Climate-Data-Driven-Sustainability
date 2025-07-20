# 🌱 Food & Climate: Data-Driven Sustainability  
**Analyzing food production, greenhouse gas emissions, and climate anomalies to shape a more sustainable future.**  
🔬 *Created as part of the AllWomen in Data Analytics Bootcamp*

---

## 📊 1. Project Overview

This project explores how global food systems contribute to climate change using open datasets spanning 1961–2022. By integrating data on food production, GHG emissions, and temperature anomalies, it reveals how agricultural trends correlate with environmental outcomes.

Over the last century, population growth (projected to reach 10 billion by 2050) has intensified demand for food, energy, and water. While advances in technology have helped meet these needs, agriculture and livestock now account for **25–30% of global CO₂ emissions**, posing serious challenges for climate resilience.

---

## 🎯 2. Key Performance Indicators (KPIs)

- 📦 Total Global Food Production (by weight)  
- 🌫️ GHG Emissions per Food Category/Product (total & per kg)  
- 🗺️ Country-Level GHG Emissions from Food Production  
- 🌡️ Correlation between Food Production/Emissions and Temperature Anomalies  

---

## ❓ 3. Research Questions

### 🔍 Food Production Trends
- How has food and feed production evolved over time?
- Which countries dominate in output and climate impact?
- What food products are most heavily produced?

### 🔎 Environmental Impact & Temperature
- Which foods contribute the most to greenhouse gas emissions?
- How have global surface temperatures changed since 1961?
- Is there a measurable relationship between emissions and temperature anomalies?

---

## 🔬 4. Hypothesis Testing

To evaluate emission differences across food categories:

- **Null Hypothesis (H₀):** There is no significant difference between the mean emissions from meat & dairy products vs. other categories.  
- **Alternative Hypothesis (H₁):** Meat & dairy generate significantly higher GHG emissions than plant-based foods.

**Test:** Z-test for independent samples  
- **Z-statistic:** −3.750  
- **p-value:** 0.00135  
- **α =** 0.05  
✅ **Result:** Rejected H₀. Meat & dairy have **statistically higher emissions**, confirming their disproportionate climate impact.

---

## 📚 5. Data Sources

Datasets sourced from Kaggle & FAOSTAT:

- 🌾 **FAOSTAT Climate Change — Agrifood Systems Emissions**  
  [FAO Dataset](http://www.fao.org/faostat/en/#data/EI)  

- 🌡️ **Temperature Change Per Country (1961–2023)**  
  [Kaggle Dataset](https://www.kaggle.com/datasets/sevgisarac/temperature-change)

---

## 🛠️ 6. Technologies & Tools

| Category              | Tools                                   |
|-----------------------|------------------------------------------|
| Data Engineering      | Python, Pandas, NumPy                    |
| Cloud & SQL           | Google BigQuery, SQL                     |
| Statistical Analysis  | SciPy, Statsmodels                       |
| Visualization         | Tableau, Matplotlib, Seaborn             |
| Data Sources          | FAOSTAT, Kaggle                          |

---

## 🔭 7. Future Work

- 🔄 Integrate population data for **per capita emissions**  
- 📊 Compare food items using the **Our World in Data API**  
- 🌍 Group countries by production/emissions scale to detect **GHG outliers**  
- 🧠 Explore **regional climate zones** to uncover localized impacts  
- ✅ Assess **GHG efficiency per food category** within country scales  

---

## 🌟 Final Insight

> “Sustainable food policy must address both the **volume of production** and the **emissions intensity** of specific food categories—while recognizing the **regional complexity** of climate change’s impact on agriculture.”

---

## 🤝 Contact & Collaboration

This project was built during the **AllWomen in Data Analytics Bootcamp** to demonstrate how open data and thoughtful analysis can fuel climate-smart innovation.

📫 **Email:** [ivanaloveraruiz@gmail.com](mailto:ivanaloveraruiz@gmail.com)

Whether you want to collaborate, fork, or contribute new angles—I’d love to connect!

---

