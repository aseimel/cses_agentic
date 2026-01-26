Micro-Processing Workflow (CSES Module 6)This document outlines the standard workflow for processing CSES Module 6 micro-level datasets from the initial deposit to the deposit of the processed dataset for cross-national phase. 

Required Input Files For Processing 

Available in the “Emails” folder on Dropbox in the respective election study’s subfolder at 'CSES Dropbox/CSES Secretariat II/CSES Data Products/CSES Standalone Modules/Module 6/Election Studies' 

* These must be provided by the respective country collaborator:
    * Data Deposit (raw survey data from national collaborator) 
    * Design Report (summary of each study’s survey design) 
    * Macro Report (summary of each election, applicable electoral system, etc.) 
    * Questionnaires as fielded in original language (required) & English back-translation (if available) 
    * Optional: Any additional files that might assist with processing (collaborator notes on provided dataset, dataset codebooks, etc.) 

To be downloaded from the CSES Dropbox before processing: 

All templates are available at 'CSES Dropbox/CSES Secretariat II/CSES Data Products/CSES Standalone Modules/Module 6/Election Studies/z) COUNTRY_YEAR_M6' – always grab the most recent copy from Dropbox! 

* Stata Syntax Templates for Processing (main do-file and check files) 
* Log File Template 
* Variable Tracking Sheet Template 
* Label Files - Micro 

Shared document with team to be updated throughout processing: 

* Release Tracking Sheet (tracks each study’s status) 
* CSES Module 6 Codebook (request ownership before updating!) 



Output Files 

* Processed dataset (in .dta format) 
* Edited Stata syntax (country-specific do-file) 
* Results of data checks (.smcl Stata logfiles) 
* Edited Log file (with details on issues faced & all required documentation) 
* Collaborator questions 
* Updated Codebook (with entries for processed election study) 



Processing Steps 

Step 0. Set Up Country Folder on Your Local Hard Drive 

* Copy the latest version of z) COUNTRY_YEAR_M6 from Dropbox to a local directory of your choice. The folder includes all required templates and is available at '/ CSES Dropbox/CSES Secretariat II/CSES Data Products/CSES Standalone Modules/Module 6/Election Studies' 
* Rename the local copy of COUNTRY_YEAR_M6 on your local drive (e.g. Germany_2025 if you’re assigned the German 2025 study). 
* Copy the deposited data file & documentation to the micro subfolder within this location. 


Step 1. Check Completeness of Deposit 

* Review deposited survey data and documentation 
* Mark provided material (and missing items) in the release tracking sheet on Dropbox at '/CSES Dropbox/CSES Secretariat II/CSES Event and Staff Calendar & Product & Release Tracker/Product & Release Tracking' 


Step 2. Read Design Report 

* Review the Design Report thoroughly 
* Verify study matches CSES standards 
* Note any questions you have on the study’s design in the logfile for later reference when drafting collaborator questions 

Check guideline at '/ CSES Dropbox/CSES Secretariat II/CSES Guidelines & Policies/CSES - Study Eligibility Checklist' for assistance 
Notify the project manager if the study does not meet methodological standards. 

Step 3. Fill Variable Tracking Sheet 

* Check variable list against CSES requirements 
* Mark deposited, missing, and unusually coded variables in Variable Tracking Sheet 

Notify project manager of any survey variables (F3XXX) expected to be included in the survey, but which were not fielded (missing F2XXX variables are usually okay). 
 
In case steps 1-3 did not result in any major issues that might impede processing: 

Step 4. Write Study Design & Weights Overview 

* Document study design based on Design Report in Logfile 
* Document weighting methodology based on Design Report in Logfile 


Step 5. Request Election Results Table from Macro Coder 

* Contact macro coder for election results table for party ordering 


Step 6. Run Frequencies on Original Data 

* Open main do-file in Stata 
* Update use path to point to deposited data 
* Update log using path 
* Run frequency section to see available variables 


Step 7. Process Variables in Stata 

* Work through do-file from top to bottom 
* For each CSES variable (F1001, F1004, F2001, etc.): 

- Find matching variable in deposited data (reference variable tracking sheet created in step 3 and saved frequencies from step 6 to assist you) 
- Write gen and recode commands 
- Run tab to check results after you have processed a variable 

* Document all issues faced in log file, how you solved these issues, and whether collaborator questions were required 


Step 8. Complete Documentation (Usually alongside Processing in Step 7) 

* Finalize log file with all processing decisions 
* Fill in codebook entries (Election Study Notes) 
* Create party & leaders table with all numerical party codes used for coding vote choice, party ID, etc., once you agreed with the macro coder on party ordering. For a guideline on how party codes are assigned, see ‘CSES Dropbox/CSES Secretariat II/CSES Guidelines & Policies/CSES - Coding of Party_Coalition & Leader_Classification Schemes & Documentation' 


Step 9. Collect and Integrate District Data 
If the respective study includes district IDs… 

* Reference macro report for a source on district-level data 
* Alternatively, identify a source for district-level *official* election results (see District Data Training for common sources) 
* Collect district data from the source (e.g., with assistance from an RA) 
* Merge the collected district data to micro dataset & process it in the micro processing syntax 

If the respective study does NOT include district IDs… 

* Check whether the polity operates a nationwide district, and process nationwide election results, if applicable 
* If the polity does NOT operate a nationwide district and district IDs are NOT provided, ask a collaborator question whether district IDs are unavailable and if so, why 


Step 10. Update Stata Label Files for Numerical Party Codes 

* Update the numeric party code labels for country-specific parties in the appropriate label templates after you have finished processing all numerical party code variables 


Step 11. Finish data processing in micro processing syntax 

* Drop all original unprocessed variables provided by collaborators such that only processed CSES variables remain in the dataset 
* Apply labels to all processed variables - do-file calls label files automatically 
* Run frequencies on the processed micro data and save them in a Stata logfile 
* Save processed micro data (Update save path with country and date) 
* Go through the logfile with frequencies on all processed variables and check for blatant errors (e.g., wild codes should pop out). 


Step 12. Run Check Files 

* Update paths in check files 
* Run inconsistency checks 
* Run theoretical checks – investigate anomalies 
* Run interview(er) validation checks 
* Save logfile outputs from all checks, review them and check whether results require any follow-ups with collaborators (draft collaborator questions as required) 


Step 13. Write Up Collaborator Questions 

* Compile clarification questions and send to project manager for review once you’re done 


Step 14. Follow Up on Collaborator Questions Upon Response 

* Track responses from collaborators 
* Update syntax and documentation as required based on provided clarifications 
* Draft up any follow-up/ vetting questions, if applicable 


Step 15. Transfer ESNs to Codebook 

* Once processing is completed, transfer all Election Study Notes (ESNs) from log file to codebook – don’t forget to request Codebook ownership first! 


Step 16. Immediately ahead of Cross-National Phase: 

* Copy the FINAL version of your processed dataset to the designated location on Dropbox (for M6 AR2: '/ CSES Dropbox/CSES Secretariat II/CSES Data Products/CSES Standalone Modules/Module 6/CSES6 AR2/Cross-National Phase - Files/Micro_data') 
* Provide the FINAL version of the Design Report and Questionnaires to the designated location on Dropbox (for M6 AR2: 'CSES Dropbox/CSES Secretariat II/CSES Data Products/CSES Standalone Modules/Module 6/CSES6 AR2/Release Files & Documentation and Other Website Updates/Accompanying Files For Release') 
* Email Project Manager to formally sign-off processing. Email should include the N, vetting questions (if applicable), and a confirmation that you finished processing (data complete, all documentation integrated into Codebook, etc.). 

Key Rules 

* Never modify shared label files – used across all countries 
* Always date filenames – for version control 
* Document everything – record all decisions and issues in log file 

