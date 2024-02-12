#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on Sun Feb 11 11:44:54 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '4'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'ESST'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/kallen3/Desktop/researchProjects/N-ACT/Measures/Behavioral/Assessment/emotionalStopSignalTask/esst/ESST v2.0.0_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1440, 900], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='myMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "loadStims" ---
    # Run 'Begin Experiment' code from codeLoadStims
    # Prepare a dictionary to hold all our stims, categorized by stim_type
    all_stims = {'ambiguous_image': [], 'negative_image': [], 'positive_image': []}
    
    # Code Component - Begin Experiment
    stimuli_list = data.importConditions('image_stims.xlsx')
    
    # You can iterate over the stimuli_list to categorize them into the dictionary if needed
    for stim in stimuli_list:
        all_stims[stim['stim_type']].append(stim['stim_file'])
        
    # Initialize counter for each stim_type at 0
    next_stim = {}
    for stim_type in all_stims.keys():
        next_stim[stim_type] = 0
    
    # --- Initialize components for Routine "shuffleStims" ---
    
    # --- Initialize components for Routine "prepTimings" ---
    # Run 'Begin Experiment' code from codePrepTimings
    from psychopy import core
    globalClock = core.Clock()  # to track the time since experiment started
    
    
    # --- Initialize components for Routine "instructions" ---
    textInstructions = visual.TextStim(win=win, name='textInstructions',
        text='',
        font='Arial',
        pos=(-0.00, -0.00), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=0.0);
    keyInstruct = keyboard.Keyboard()
    # Run 'Begin Experiment' code from codeInstructions
    # Code components should usually appear at the top
    # of the routine. This one has to appear after the
    # text component it refers to.
    textInstructions.alignText='left'
    
    
    # --- Initialize components for Routine "prepPrac" ---
    textPractice = visual.TextStim(win=win, name='textPractice',
        text='',
        font='Arial',
        pos=(-0, -0), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    keyPractice = keyboard.Keyboard()
    
    # --- Initialize components for Routine "prepTrial" ---
    
    # --- Initialize components for Routine "jitteredISI" ---
    textJitteredFix = visual.TextStim(win=win, name='textJitteredFix',
        text='+',
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "preTrial" ---
    textPreTrialFix = visual.TextStim(win=win, name='textPreTrialFix',
        text='+',
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "iceTrial" ---
    imageTarg = visual.ImageStim(
        win=win,
        name='imageTarg', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    keyResp = keyboard.Keyboard()
    stopSignal = sound.Sound('A', secs=.5, stereo=True, hamming=True,
        name='stopSignal')
    stopSignal.setVolume(1.0)
    
    # --- Initialize components for Routine "postTrial" ---
    textPostTrial = visual.TextStim(win=win, name='textPostTrial',
        text='',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "blockFeedback" ---
    blockFeedbackText = visual.TextStim(win=win, name='blockFeedbackText',
        text='',
        font='Arial',
        pos=(0, 0), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "endTask" ---
    textEnd = visual.TextStim(win=win, name='textEnd',
        text="Well done! You've finished this test. Thank you for participating in our study.\n\nPlease press the <SPACEBAR> to end the experiment and inform a research team member.",
        font='Arial',
        pos=(0, 0), height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_respEnd = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # set up handler to look after randomisation of conditions etc
    imgLoader = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('image_stims.xlsx'),
        seed=None, name='imgLoader')
    thisExp.addLoop(imgLoader)  # add the loop to the experiment
    thisImgLoader = imgLoader.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisImgLoader.rgb)
    if thisImgLoader != None:
        for paramName in thisImgLoader:
            globals()[paramName] = thisImgLoader[paramName]
    
    for thisImgLoader in imgLoader:
        currentLoop = imgLoader
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisImgLoader.rgb)
        if thisImgLoader != None:
            for paramName in thisImgLoader:
                globals()[paramName] = thisImgLoader[paramName]
        
        # --- Prepare to start Routine "loadStims" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('loadStims.started', globalClock.getTime())
        # keep track of which components have finished
        loadStimsComponents = []
        for thisComponent in loadStimsComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "loadStims" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in loadStimsComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "loadStims" ---
        for thisComponent in loadStimsComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('loadStims.stopped', globalClock.getTime())
        # the Routine "loadStims" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 1.0 repeats of 'imgLoader'
    
    
    # --- Prepare to start Routine "shuffleStims" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('shuffleStims.started', globalClock.getTime())
    # Run 'Begin Routine' code from codeShuffleStims
    from random import shuffle
    # Shuffle the list of stims for each stim type directly
    for key in all_stims:
        shuffle(all_stims[key]) 
    #create a new_stim dict
    next_stim = {}
    next_stim[key+"_"] = 0
    # keep track of which components have finished
    shuffleStimsComponents = []
    for thisComponent in shuffleStimsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "shuffleStims" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in shuffleStimsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "shuffleStims" ---
    for thisComponent in shuffleStimsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('shuffleStims.stopped', globalClock.getTime())
    # the Routine "shuffleStims" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "prepTimings" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('prepTimings.started', globalClock.getTime())
    # Run 'Begin Routine' code from codePrepTimings
    #set to 'true' for debugging
    debug_mode = False
    
    #set the buzz timer params
    buzz_min = 0.25
    buzz_max = 1.45
    buzz_default = 0.5
    
    #create a dictionary to hold all of our stop-signal delay (SSD) values
    buzz_timers = {}
    
    go_accuracies = []
    go_reaction_times = []
    stop_accuracies = []
    stop_reaction_times = []
    
    #get a list of all the stim types
    key_list = list(all_stims)
    
    #for each stim category...
    for key in key_list:
        buzz_timers[key] = buzz_default
    # keep track of which components have finished
    prepTimingsComponents = []
    for thisComponent in prepTimingsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "prepTimings" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prepTimingsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prepTimings" ---
    for thisComponent in prepTimingsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('prepTimings.stopped', globalClock.getTime())
    # the Routine "prepTimings" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions.started', globalClock.getTime())
    keyInstruct.keys = []
    keyInstruct.rt = []
    _keyInstruct_allKeys = []
    # keep track of which components have finished
    instructionsComponents = [textInstructions, keyInstruct]
    for thisComponent in instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textInstructions* updates
        
        # if textInstructions is starting this frame...
        if textInstructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textInstructions.frameNStart = frameN  # exact frame index
            textInstructions.tStart = t  # local t and not account for scr refresh
            textInstructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textInstructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textInstructions.started')
            # update status
            textInstructions.status = STARTED
            textInstructions.setAutoDraw(True)
        
        # if textInstructions is active this frame...
        if textInstructions.status == STARTED:
            # update params
            textInstructions.setText("This is a timed test.\n\nOn each trial, you will see an image.\n\nYour task is to decide whether each image is POSITIVE or NEGATIVE:\n\n- Press the 'A' key (marked with a smiling sticker) if the image is pleasant or POSITIVE.\n\n- Press the 'L' key (marked with a frowning sticker) if the image is unpleasant or NEGATIVE. \n\nIf you are unsure how a picture makes you feel, follow your instinct or 'gut reaction'. \n\nPlease respond as quickly and accurately as possible!\n\nHowever, if you hear a sound (called the 'STOP SIGNAL'), you should STOP your response on that trial. DO NOT respond to the image when you hear this sound.\n\nYou should stop your response regardless of whether the image has just appeared or if the STOP SIGNAL plays afterwards. Nevertheless, DO NOT WAIT for the STOP SIGNAL. If you wait, the test will get harder on later trials.\n\nPress <SPACEBAR> to practice.", log=False)
        
        # *keyInstruct* updates
        waitOnFlip = False
        
        # if keyInstruct is starting this frame...
        if keyInstruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            keyInstruct.frameNStart = frameN  # exact frame index
            keyInstruct.tStart = t  # local t and not account for scr refresh
            keyInstruct.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(keyInstruct, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'keyInstruct.started')
            # update status
            keyInstruct.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(keyInstruct.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(keyInstruct.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if keyInstruct.status == STARTED and not waitOnFlip:
            theseKeys = keyInstruct.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _keyInstruct_allKeys.extend(theseKeys)
            if len(_keyInstruct_allKeys):
                keyInstruct.keys = _keyInstruct_allKeys[-1].name  # just the last key pressed
                keyInstruct.rt = _keyInstruct_allKeys[-1].rt
                keyInstruct.duration = _keyInstruct_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions.stopped', globalClock.getTime())
    # check responses
    if keyInstruct.keys in ['', [], None]:  # No response was made
        keyInstruct.keys = None
    thisExp.addData('keyInstruct.keys',keyInstruct.keys)
    if keyInstruct.keys != None:  # we had a response
        thisExp.addData('keyInstruct.rt', keyInstruct.rt)
        thisExp.addData('keyInstruct.duration', keyInstruct.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    pracBlock_loop = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='pracBlock_loop')
    thisExp.addLoop(pracBlock_loop)  # add the loop to the experiment
    thisPracBlock_loop = pracBlock_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPracBlock_loop.rgb)
    if thisPracBlock_loop != None:
        for paramName in thisPracBlock_loop:
            globals()[paramName] = thisPracBlock_loop[paramName]
    
    for thisPracBlock_loop in pracBlock_loop:
        currentLoop = pracBlock_loop
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPracBlock_loop.rgb)
        if thisPracBlock_loop != None:
            for paramName in thisPracBlock_loop:
                globals()[paramName] = thisPracBlock_loop[paramName]
        
        # --- Prepare to start Routine "prepPrac" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('prepPrac.started', globalClock.getTime())
        # Run 'Begin Routine' code from codePrepPrac
        #print("block_loop.thisN")
        #print(block_loop.thisN)
        block_title = "Practice"
        textPractice.alignText='left'
        
        '''
        if(block_loop.thisN == 0):
            p_cue_target_match["AA"] = 1
            p_cue_target_match["NA"] = 1
            p_cue_target_match["PA"] = 1
        
            p_cue_target_match["AN"] = 1
            p_cue_target_match["NN"] = 1
            p_cue_target_match["PN"] = 1
        
            p_cue_target_match["AP"] = 1
            p_cue_target_match["NP"] = 1
            p_cue_target_match["PP"] = 1
        
            p_signal_to_noise["AA"] = 1
            p_signal_to_noise["AN"] = 1
            p_signal_to_noise["AP"] = 1
        
            p_signal_to_noise["NA"] = 1
            p_signal_to_noise["NN"] = 1
            p_signal_to_noise["NP"] = 1
        
            p_signal_to_noise["PA"] = 1
            p_signal_to_noise["PN"] = 1
            p_signal_to_noise["PP"] = 1
            
        elif(block_loop.thisN == 1):
            
            p_cue_target_match["AA"] = .7
            p_cue_target_match["NA"] = .7
            p_cue_target_match["PA"] = .7
        
            p_cue_target_match["AN"] = .7
            p_cue_target_match["NN"] = .7
            p_cue_target_match["PN"] = .7
        
            p_cue_target_match["AP"] = .7
            p_cue_target_match["NP"] = .7
            p_cue_target_match["PP"] = .7
        
            p_signal_to_noise["AA"] = .7
            p_signal_to_noise["AN"] = .7
            p_signal_to_noise["AP"] = .7
        
            p_signal_to_noise["NA"] = .7
            p_signal_to_noise["NN"] = .7
            p_signal_to_noise["NP"] = .7
        
            p_signal_to_noise["PA"] = .7
            p_signal_to_noise["PN"] = .7
            p_signal_to_noise["PP"] = .7
        '''
        keyPractice.keys = []
        keyPractice.rt = []
        _keyPractice_allKeys = []
        # keep track of which components have finished
        prepPracComponents = [textPractice, keyPractice]
        for thisComponent in prepPracComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prepPrac" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textPractice* updates
            
            # if textPractice is starting this frame...
            if textPractice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textPractice.frameNStart = frameN  # exact frame index
                textPractice.tStart = t  # local t and not account for scr refresh
                textPractice.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textPractice, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textPractice.started')
                # update status
                textPractice.status = STARTED
                textPractice.setAutoDraw(True)
            
            # if textPractice is active this frame...
            if textPractice.status == STARTED:
                # update params
                textPractice.setText('First, some practice. Press <SPACEBAR> to start.', log=False)
            
            # *keyPractice* updates
            waitOnFlip = False
            
            # if keyPractice is starting this frame...
            if keyPractice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                keyPractice.frameNStart = frameN  # exact frame index
                keyPractice.tStart = t  # local t and not account for scr refresh
                keyPractice.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(keyPractice, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'keyPractice.started')
                # update status
                keyPractice.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(keyPractice.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(keyPractice.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if keyPractice.status == STARTED and not waitOnFlip:
                theseKeys = keyPractice.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _keyPractice_allKeys.extend(theseKeys)
                if len(_keyPractice_allKeys):
                    keyPractice.keys = _keyPractice_allKeys[-1].name  # just the last key pressed
                    keyPractice.rt = _keyPractice_allKeys[-1].rt
                    keyPractice.duration = _keyPractice_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prepPracComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prepPrac" ---
        for thisComponent in prepPracComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('prepPrac.stopped', globalClock.getTime())
        # check responses
        if keyPractice.keys in ['', [], None]:  # No response was made
            keyPractice.keys = None
        pracBlock_loop.addData('keyPractice.keys',keyPractice.keys)
        if keyPractice.keys != None:  # we had a response
            pracBlock_loop.addData('keyPractice.rt', keyPractice.rt)
            pracBlock_loop.addData('keyPractice.duration', keyPractice.duration)
        # the Routine "prepPrac" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        pracTrials_loop = data.TrialHandler(nReps=1.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('block_pracTrials.xlsx'),
            seed=None, name='pracTrials_loop')
        thisExp.addLoop(pracTrials_loop)  # add the loop to the experiment
        thisPracTrials_loop = pracTrials_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPracTrials_loop.rgb)
        if thisPracTrials_loop != None:
            for paramName in thisPracTrials_loop:
                globals()[paramName] = thisPracTrials_loop[paramName]
        
        for thisPracTrials_loop in pracTrials_loop:
            currentLoop = pracTrials_loop
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisPracTrials_loop.rgb)
            if thisPracTrials_loop != None:
                for paramName in thisPracTrials_loop:
                    globals()[paramName] = thisPracTrials_loop[paramName]
            
            # --- Prepare to start Routine "prepTrial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('prepTrial.started', globalClock.getTime())
            # Run 'Begin Routine' code from codePrepTrial
            import random
            from random import shuffle
            
            # Sort out the jitter
            jitter_times = [0.150, 0.200, 0.250, 0.300, 0.350]
            shuffle(jitter_times)
            jitter_time = jitter_times[0]
            thisExp.addData("jitter_time", jitter_time)
            
            # Select the next available stim for the given stim_type
            if next_stim[stim_type] >= len(all_stims[stim_type]):
                next_stim[stim_type] = 0  # Reset the counter if exceeded
            
            stimulus_image = all_stims[stim_type][next_stim[stim_type]]
            next_stim[stim_type] += 1  # Increment the next stim counter
            
            thisExp.addData("stimulus_image", stimulus_image)
            
            # Select the trial time
            trial_time = 1.25
            thisExp.addData("trial_time", trial_time)
            
            buzz_time = 0
            buzz_volume = 0
            
            # Setup the sound for non-go trials
            if go_nogo != "go":
                buzz_time = buzz_timers[stim_type]
                buzz_volume = 1
                thisExp.addData("buzz_time", buzz_time)
            # keep track of which components have finished
            prepTrialComponents = []
            for thisComponent in prepTrialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prepTrial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prepTrialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prepTrial" ---
            for thisComponent in prepTrialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('prepTrial.stopped', globalClock.getTime())
            # the Routine "prepTrial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "jitteredISI" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('jitteredISI.started', globalClock.getTime())
            # keep track of which components have finished
            jitteredISIComponents = [textJitteredFix]
            for thisComponent in jitteredISIComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "jitteredISI" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *textJitteredFix* updates
                
                # if textJitteredFix is starting this frame...
                if textJitteredFix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textJitteredFix.frameNStart = frameN  # exact frame index
                    textJitteredFix.tStart = t  # local t and not account for scr refresh
                    textJitteredFix.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textJitteredFix, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textJitteredFix.started')
                    # update status
                    textJitteredFix.status = STARTED
                    textJitteredFix.setAutoDraw(True)
                
                # if textJitteredFix is active this frame...
                if textJitteredFix.status == STARTED:
                    # update params
                    pass
                
                # if textJitteredFix is stopping this frame...
                if textJitteredFix.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > textJitteredFix.tStartRefresh + jitter_time-frameTolerance:
                        # keep track of stop time/frame for later
                        textJitteredFix.tStop = t  # not accounting for scr refresh
                        textJitteredFix.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'textJitteredFix.stopped')
                        # update status
                        textJitteredFix.status = FINISHED
                        textJitteredFix.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in jitteredISIComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "jitteredISI" ---
            for thisComponent in jitteredISIComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('jitteredISI.stopped', globalClock.getTime())
            # the Routine "jitteredISI" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "preTrial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('preTrial.started', globalClock.getTime())
            # keep track of which components have finished
            preTrialComponents = [textPreTrialFix]
            for thisComponent in preTrialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "preTrial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *textPreTrialFix* updates
                
                # if textPreTrialFix is starting this frame...
                if textPreTrialFix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textPreTrialFix.frameNStart = frameN  # exact frame index
                    textPreTrialFix.tStart = t  # local t and not account for scr refresh
                    textPreTrialFix.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textPreTrialFix, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textPreTrialFix.started')
                    # update status
                    textPreTrialFix.status = STARTED
                    textPreTrialFix.setAutoDraw(True)
                
                # if textPreTrialFix is active this frame...
                if textPreTrialFix.status == STARTED:
                    # update params
                    pass
                
                # if textPreTrialFix is stopping this frame...
                if textPreTrialFix.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > textPreTrialFix.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        textPreTrialFix.tStop = t  # not accounting for scr refresh
                        textPreTrialFix.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'textPreTrialFix.stopped')
                        # update status
                        textPreTrialFix.status = FINISHED
                        textPreTrialFix.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in preTrialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "preTrial" ---
            for thisComponent in preTrialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('preTrial.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.500000)
            
            # --- Prepare to start Routine "iceTrial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('iceTrial.started', globalClock.getTime())
            imageTarg.setPos((0, 0))
            imageTarg.setImage(stimulus_image)
            keyResp.keys = []
            keyResp.rt = []
            _keyResp_allKeys = []
            stopSignal.setSound('A', secs=.5, hamming=True)
            stopSignal.setVolume(buzz_volume, log=False)
            stopSignal.seek(0)
            # keep track of which components have finished
            iceTrialComponents = [imageTarg, keyResp, stopSignal]
            for thisComponent in iceTrialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "iceTrial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *imageTarg* updates
                
                # if imageTarg is starting this frame...
                if imageTarg.status == NOT_STARTED and tThisFlip >= 0.25-frameTolerance:
                    # keep track of start time/frame for later
                    imageTarg.frameNStart = frameN  # exact frame index
                    imageTarg.tStart = t  # local t and not account for scr refresh
                    imageTarg.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(imageTarg, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'imageTarg.started')
                    # update status
                    imageTarg.status = STARTED
                    imageTarg.setAutoDraw(True)
                
                # if imageTarg is active this frame...
                if imageTarg.status == STARTED:
                    # update params
                    pass
                
                # if imageTarg is stopping this frame...
                if imageTarg.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > imageTarg.tStartRefresh + trial_time-frameTolerance:
                        # keep track of stop time/frame for later
                        imageTarg.tStop = t  # not accounting for scr refresh
                        imageTarg.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'imageTarg.stopped')
                        # update status
                        imageTarg.status = FINISHED
                        imageTarg.setAutoDraw(False)
                
                # *keyResp* updates
                waitOnFlip = False
                
                # if keyResp is starting this frame...
                if keyResp.status == NOT_STARTED and tThisFlip >= 0.25-frameTolerance:
                    # keep track of start time/frame for later
                    keyResp.frameNStart = frameN  # exact frame index
                    keyResp.tStart = t  # local t and not account for scr refresh
                    keyResp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(keyResp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'keyResp.started')
                    # update status
                    keyResp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(keyResp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(keyResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if keyResp is stopping this frame...
                if keyResp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > keyResp.tStartRefresh + trial_time-frameTolerance:
                        # keep track of stop time/frame for later
                        keyResp.tStop = t  # not accounting for scr refresh
                        keyResp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'keyResp.stopped')
                        # update status
                        keyResp.status = FINISHED
                        keyResp.status = FINISHED
                if keyResp.status == STARTED and not waitOnFlip:
                    theseKeys = keyResp.getKeys(keyList=['a','l'], ignoreKeys=["escape"], waitRelease=False)
                    _keyResp_allKeys.extend(theseKeys)
                    if len(_keyResp_allKeys):
                        keyResp.keys = _keyResp_allKeys[-1].name  # just the last key pressed
                        keyResp.rt = _keyResp_allKeys[-1].rt
                        keyResp.duration = _keyResp_allKeys[-1].duration
                        # was this correct?
                        if (keyResp.keys == str('')) or (keyResp.keys == ''):
                            keyResp.corr = 1
                        else:
                            keyResp.corr = 0
                
                # if stopSignal is starting this frame...
                if stopSignal.status == NOT_STARTED and tThisFlip >= buzz_time-frameTolerance:
                    # keep track of start time/frame for later
                    stopSignal.frameNStart = frameN  # exact frame index
                    stopSignal.tStart = t  # local t and not account for scr refresh
                    stopSignal.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('stopSignal.started', tThisFlipGlobal)
                    # update status
                    stopSignal.status = STARTED
                    stopSignal.play(when=win)  # sync with win flip
                
                # if stopSignal is stopping this frame...
                if stopSignal.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > stopSignal.tStartRefresh + .5-frameTolerance:
                        # keep track of stop time/frame for later
                        stopSignal.tStop = t  # not accounting for scr refresh
                        stopSignal.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'stopSignal.stopped')
                        # update status
                        stopSignal.status = FINISHED
                        stopSignal.stop()
                # update stopSignal status according to whether it's playing
                if stopSignal.isPlaying:
                    stopSignal.status = STARTED
                elif stopSignal.isFinished:
                    stopSignal.status = FINISHED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in iceTrialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "iceTrial" ---
            for thisComponent in iceTrialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('iceTrial.stopped', globalClock.getTime())
            # check responses
            if keyResp.keys in ['', [], None]:  # No response was made
                keyResp.keys = None
                # was no response the correct answer?!
                if str('').lower() == 'none':
                   keyResp.corr = 1;  # correct non-response
                else:
                   keyResp.corr = 0;  # failed to respond (incorrectly)
            # store data for pracTrials_loop (TrialHandler)
            pracTrials_loop.addData('keyResp.keys',keyResp.keys)
            pracTrials_loop.addData('keyResp.corr', keyResp.corr)
            if keyResp.keys != None:  # we had a response
                pracTrials_loop.addData('keyResp.rt', keyResp.rt)
                pracTrials_loop.addData('keyResp.duration', keyResp.duration)
            stopSignal.pause()  # ensure sound has stopped at end of Routine
            # Run 'End Routine' code from codeTrial
            # Default feedback text in case there's an error or no response component found
            feedback_message = 'No key_resp component found - look at the Std out window for info'
            fb_col = 'black'
            
            if debug_mode:
                # (Your debug mode code remains the same)
                ...
            else:
                # Handling responses for both go and no-go trials
                if go_nogo != "go":
                    # Handling for no-go trials
                    if keyResp.keys is not None:  # If a response was made
                        feedback_message = "False alarm! Please do not respond on stop trials."
                        fb_col = "red"
                        keyResp.corr = 0
                        # Decrease buzz_time
                        buzz_time = max(buzz_min, buzz_time - 0.05)
                    else:  # If no response was made
                        feedback_message = "Correct! No response needed on stop trials."
                        fb_col = "green"
                        keyResp.corr = 1
                        # Increase buzz_time
                        buzz_time = min(buzz_max, buzz_time + 0.05)
            
                    # Common updates for no-go trials
                    thisExp.addData("no_go_acc", keyResp.corr)
                    stop_accuracies.append(keyResp.corr)
                    buzz_timers[stim_type] = buzz_time
                    thisExp.addData('new_buzz_time', buzz_time)
                    
                elif go_nogo == "go":
                    # Handling for go trials
                    if keyResp.keys:  # If a response was made
                        keyResp.corr = int(keyResp.keys[0] in [corr_resp1, corr_resp2])
                        feedback_message = "Correct!" if keyResp.corr else "Oops! That was the wrong response."
                        fb_col = "green" if keyResp.corr else "red"
                    else:  # If no response was made
                        keyResp.corr = 0  # Incorrect omission error
                        feedback_message = "Miss! Please try to respond more quickly."
                        fb_col = "red"
            
                    # Common updates for go trials
                    thisExp.addData('go_reaction_time', keyResp.rt[0] if keyResp.keys else None)
                    thisExp.addData('go_accuracy', keyResp.corr)
                    go_accuracies.append(keyResp.corr)
            
                print("buzz_time for " + str(stim_type) + " is " + str(buzz_time))
            # the Routine "iceTrial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "postTrial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('postTrial.started', globalClock.getTime())
            textPostTrial.setColor(fb_col, colorSpace='rgb')
            textPostTrial.setText(feedback_message)
            # keep track of which components have finished
            postTrialComponents = [textPostTrial]
            for thisComponent in postTrialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "postTrial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *textPostTrial* updates
                
                # if textPostTrial is starting this frame...
                if textPostTrial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textPostTrial.frameNStart = frameN  # exact frame index
                    textPostTrial.tStart = t  # local t and not account for scr refresh
                    textPostTrial.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textPostTrial, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textPostTrial.started')
                    # update status
                    textPostTrial.status = STARTED
                    textPostTrial.setAutoDraw(True)
                
                # if textPostTrial is active this frame...
                if textPostTrial.status == STARTED:
                    # update params
                    pass
                
                # if textPostTrial is stopping this frame...
                if textPostTrial.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > textPostTrial.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        textPostTrial.tStop = t  # not accounting for scr refresh
                        textPostTrial.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'textPostTrial.stopped')
                        # update status
                        textPostTrial.status = FINISHED
                        textPostTrial.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in postTrialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "postTrial" ---
            for thisComponent in postTrialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('postTrial.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'pracTrials_loop'
        
        # get names of stimulus parameters
        if pracTrials_loop.trialList in ([], [None], None):
            params = []
        else:
            params = pracTrials_loop.trialList[0].keys()
        # save data for this loop
        pracTrials_loop.saveAsExcel(filename + '.xlsx', sheetName='pracTrials_loop',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        pracTrials_loop.saveAsText(filename + 'pracTrials_loop.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "blockFeedback" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('blockFeedback.started', globalClock.getTime())
        # Run 'Begin Routine' code from codeBlockFB
        ## codeBlockFB Component - Begin Routine
        
        # Initialize feedbackMsg
        feedbackMsg = ""
        
        # Calculate average accuracy and reaction time for "go" trials
        if go_accuracies:  # Ensure there are 'go' trials to calculate averages
            avg_go_accuracy = sum(go_accuracies) / len(go_accuracies)
            avg_go_reaction_time = sum(go_reaction_times) / len(go_reaction_times) if go_reaction_times else 0
        else:
            avg_go_accuracy = avg_go_reaction_time = 0  # Default values if no 'go' trials
        
        # Calculate for "stop-signal" trials
        if stop_accuracies:  # Ensure there are 'stop-signal' trials to calculate averages
            avg_stop_accuracy = sum(stop_accuracies) / len(stop_accuracies)
        else:
            avg_stop_accuracy = 0  # Default values if no 'stop-signal' trials
        
        # Initial part of the feedback message
        blockFeedbackText.alignText = 'left'
        feedbackMsg = f"Go Trials - Accuracy: {avg_go_accuracy:.2%}, "  # Converted to percentage
        feedbackMsg += f"Average Reaction Time: {avg_go_reaction_time:.3f} seconds\n"
        feedbackMsg += f"Stop-Signal Trials - Accuracy: {avg_stop_accuracy:.2%}\n"  # Converted to percentage
        
        # Initialize the countdown timer
        countdown_start = 10  # seconds
        globalClock = core.Clock()  # Assuming globalClock is defined at the experiment's start
        blockFeedbackText.setColor(fb_col, colorSpace='rgb')
        blockFeedbackText.setText(feedbackMsg)
        # keep track of which components have finished
        blockFeedbackComponents = [blockFeedbackText]
        for thisComponent in blockFeedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blockFeedback" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from codeBlockFB
            # codeBlockFB Component - Each Frame
            
            # Calculate the time left for the countdown
            time_elapsed = globalClock.getTime()
            countdown_time = countdown_start - time_elapsed
            
            # Update the feedback message with the countdown time
            feedbackMsg = f"Go Trials - Accuracy: {avg_go_accuracy:.2%}, "  # Converted to percentage
            feedbackMsg += f"Average Reaction Time: {avg_go_reaction_time:.3f} seconds\n"
            feedbackMsg += f"Stop-Signal Trials - Accuracy: {avg_stop_accuracy:.2%}\n"  # Converted to percentage
            feedbackMsg += f"Next block starts in: {int(countdown_time)} seconds"
            
            # Update the display of the countdown timer
            blockFeedbackText.setText(feedbackMsg)
            
            # End routine if countdown has finished
            if countdown_time <= 0:
                continueRoutine = False
            
            # *blockFeedbackText* updates
            
            # if blockFeedbackText is starting this frame...
            if blockFeedbackText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blockFeedbackText.frameNStart = frameN  # exact frame index
                blockFeedbackText.tStart = t  # local t and not account for scr refresh
                blockFeedbackText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blockFeedbackText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blockFeedbackText.started')
                # update status
                blockFeedbackText.status = STARTED
                blockFeedbackText.setAutoDraw(True)
            
            # if blockFeedbackText is active this frame...
            if blockFeedbackText.status == STARTED:
                # update params
                pass
            
            # if blockFeedbackText is stopping this frame...
            if blockFeedbackText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blockFeedbackText.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    blockFeedbackText.tStop = t  # not accounting for scr refresh
                    blockFeedbackText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blockFeedbackText.stopped')
                    # update status
                    blockFeedbackText.status = FINISHED
                    blockFeedbackText.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blockFeedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blockFeedback" ---
        for thisComponent in blockFeedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('blockFeedback.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'pracBlock_loop'
    
    # get names of stimulus parameters
    if pracBlock_loop.trialList in ([], [None], None):
        params = []
    else:
        params = pracBlock_loop.trialList[0].keys()
    # save data for this loop
    pracBlock_loop.saveAsExcel(filename + '.xlsx', sheetName='pracBlock_loop',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    pracBlock_loop.saveAsText(filename + 'pracBlock_loop.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "endTask" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('endTask.started', globalClock.getTime())
    key_respEnd.keys = []
    key_respEnd.rt = []
    _key_respEnd_allKeys = []
    # keep track of which components have finished
    endTaskComponents = [textEnd, key_respEnd]
    for thisComponent in endTaskComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "endTask" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textEnd* updates
        
        # if textEnd is starting this frame...
        if textEnd.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textEnd.frameNStart = frameN  # exact frame index
            textEnd.tStart = t  # local t and not account for scr refresh
            textEnd.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textEnd, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textEnd.started')
            # update status
            textEnd.status = STARTED
            textEnd.setAutoDraw(True)
        
        # if textEnd is active this frame...
        if textEnd.status == STARTED:
            # update params
            pass
        
        # *key_respEnd* updates
        waitOnFlip = False
        
        # if key_respEnd is starting this frame...
        if key_respEnd.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_respEnd.frameNStart = frameN  # exact frame index
            key_respEnd.tStart = t  # local t and not account for scr refresh
            key_respEnd.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_respEnd, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_respEnd.started')
            # update status
            key_respEnd.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_respEnd.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_respEnd.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_respEnd.status == STARTED and not waitOnFlip:
            theseKeys = key_respEnd.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_respEnd_allKeys.extend(theseKeys)
            if len(_key_respEnd_allKeys):
                key_respEnd.keys = _key_respEnd_allKeys[-1].name  # just the last key pressed
                key_respEnd.rt = _key_respEnd_allKeys[-1].rt
                key_respEnd.duration = _key_respEnd_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in endTaskComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "endTask" ---
    for thisComponent in endTaskComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('endTask.stopped', globalClock.getTime())
    # check responses
    if key_respEnd.keys in ['', [], None]:  # No response was made
        key_respEnd.keys = None
    thisExp.addData('key_respEnd.keys',key_respEnd.keys)
    if key_respEnd.keys != None:  # we had a response
        thisExp.addData('key_respEnd.rt', key_respEnd.rt)
        thisExp.addData('key_respEnd.duration', key_respEnd.duration)
    thisExp.nextEntry()
    # the Routine "endTask" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
