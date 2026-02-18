# ONET DATASET USE EXPLAINED

- Dataset Name : O*NET Database 
- Provider : US Department of Labor

##Dataset Overview
**Purpose in this project:**  

Used as a gender-neutral external knowledge source for constructing a role–trait verification knowledge graph in the RAV framework.  

## Dataset Versioning 
- Version : O*NET 30.1 Database 
- Release Date : December 2025
- Accessed : 19/01/2025  
- Access url : *https://www.onetcenter.org/database.html#wv*

All experiments in this repository rely exclusively on this O*NET release.
No files are modified prior to processing.

## Files Used  
- Datafiles used :  
    1. Occupation Data.txt: Occupation titles and codes   
    2. Skills.txt: Skill elements with importance ratings  
    3. Abilities.txt: Ability elements with importance ratings  
    4. Work Styles.txt: Work Style elements with importance ratings

## Identifier and Join Keys  
Primary join key:
O*NET-SOC Code
This identifier is used to:
Link occupations to traits
Align prompt roles with KG nodes
No alternative identifiers are used.  

## Data Integrity Policy
Raw O*NET files are stored unchanged
No rows or columns are removed
No values are edited or imputed
All filtering occurs downstream during prompt and KG construction
Derived artifacts are generated programmatically and are not treated as authoritative data sources.  

## Trait Interpretation Policy
In this project:
Trait is an umbrella term referring to:
Skills
Abilities
Work Styles
Trait importance values are taken directly from O*NET
No demographic or gender attributes are included
The dataset is treated as gender-neutral by design.  

## Limitations & Scope Notes
O*NET reflects U.S. occupational descriptions
It does not encode demographic participation rates
Trait importance ratings are aggregate and not role-contextualized by gender
These limitations are acknowledged and do not affect the role of O*NET as a neutral verification reference.  

## Reproducibility Statement
Given the O*NET release version specified above and the deterministic pipeline in this repository, the verification knowledge graph can be reconstructed exactly.