"use client";

import React, { useState, useEffect } from 'react';
import { supabase } from '@/lib/supabaseClient';
import {
  Flag, Clock, CheckCircle2, XCircle, RotateCcw,
  History, ChevronRight, ChevronLeft, Loader2
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// --- Types ---
type Question = {
  id: number;
  text: string;
  options: string[]; // Stored as JSONB in Supabase
  correct_answer: number;
};

type QuizHistory = {
  date: string;
  score: number;
  total: number;
};

// --- Utils ---
function cn(...inputs: ClassValue[]) { return twMerge(clsx(inputs)); }

export default function QuizApp() {
  // --- Global State ---
  const [loading, setLoading] = useState(false);
  const [activeQuestions, setActiveQuestions] = useState<Question[]>([]);
  const [quizState, setQuizState] = useState<'menu' | 'playing' | 'review'>('menu');

  // --- Session State ---
  const [currentIndex, setCurrentIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState<Record<number, number>>({});
  const [flagged, setFlagged] = useState<Set<number>>(new Set());
  const [timeLeft, setTimeLeft] = useState<number | null>(null);

  // --- Settings & History ---
  const [useTimer, setUseTimer] = useState(false);
  const [history, setHistory] = useState<QuizHistory[]>([]);

  // --- Add Question Form ---
  const [showAddForm, setShowAddForm] = useState(false);
  const [newQText, setNewQText] = useState("");
  const [newQOptions, setNewQOptions] = useState(["", ""]);
  const [newQCorrect, setNewQCorrect] = useState(0);
  const [submitStatus, setSubmitStatus] = useState("");

  // --- Initial Load ---
  useEffect(() => {
    const storedHistory = localStorage.getItem('quizHistory');
    if (storedHistory) setHistory(JSON.parse(storedHistory));
  }, []);

  // --- 1. FETCH QUESTIONS (Random 40) ---
  const startQuiz = async () => {
    setLoading(true);

    // Supabase doesn't have a native "RANDOM()" in the JS client easily,
    // so for <1000 questions, fetching ID list is fast, then we pick random IDs.
    const { data, error } = await supabase
      .from('questions')
      .select('*');

    if (error) {
      alert("Error fetching questions: " + error.message);
      setLoading(false);
      return;
    }

    if (!data || data.length === 0) {
      alert("No questions found in database! Add some first.");
      setLoading(false);
      return;
    }

    // Shuffle and slice 40
    const shuffled = data.sort(() => 0.5 - Math.random());
    const selected = shuffled.slice(0, 40);

    setActiveQuestions(selected);
    setCurrentIndex(0);
    setUserAnswers({});
    setFlagged(new Set());
    setQuizState('playing');
    setTimeLeft(useTimer ? 30 * 60 : null);
    setLoading(false);
  };

  // --- 2. TIMER LOGIC ---
  useEffect(() => {
    if (quizState !== 'playing' || timeLeft === null) return;
    if (timeLeft === 0) { finishQuiz(); return; }
    const timerId = setInterval(() => setTimeLeft(p => (p !== null ? p - 1 : null)), 1000);
    return () => clearInterval(timerId);
  }, [timeLeft, quizState]);

  // --- 3. SUBMIT NEW QUESTION ---
  const handleAddQuestion = async () => {
    if (!newQText || newQOptions.some(o => !o)) return;
    setSubmitStatus("Submitting...");

    const { error } = await supabase
      .from('questions')
      .insert([{
        text: newQText,
        options: newQOptions,
        correct_answer: newQCorrect
      }]);

    if (error) {
      setSubmitStatus("Error: " + error.message);
    } else {
      setSubmitStatus("Question Added Successfully!");
      setNewQText("");
      setNewQOptions(["", ""]);
      setNewQCorrect(0);
      setTimeout(() => {
        setShowAddForm(false);
        setSubmitStatus("");
      }, 1500);
    }
  };

  // --- 4. FINISH & HISTORY ---
  const finishQuiz = () => {
    setQuizState('review');
    let correct = 0;
    activeQuestions.forEach(q => {
      if (userAnswers[q.id] === q.correct_answer) correct++;
    });

    const newEntry = {
      date: new Date().toLocaleDateString(),
      score: correct,
      total: activeQuestions.length
    };
    const updatedHistory = [newEntry, ...history].slice(0, 10);
    setHistory(updatedHistory);
    localStorage.setItem('quizHistory', JSON.stringify(updatedHistory));
  };

  // --- RENDER HELPERS ---
  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  // --- VIEW: MENU ---
  if (quizState === 'menu') {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4 font-sans text-slate-800">
        <div className="max-w-2xl w-full bg-white rounded-2xl shadow-xl p-8">
          <h1 className="text-4xl font-bold text-slate-900 mb-2">QuizMaster Pro</h1>
          <p className="text-slate-500 mb-8">Community Driven Exam Prep</p>

          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-6">
              <div className="bg-blue-50 p-6 rounded-xl border border-blue-100">
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  <RotateCcw className="w-5 h-5 text-blue-600" /> Start New Quiz
                </h2>
                <div className="flex items-center gap-3 mb-6">
                  <input type="checkbox" id="timer" checked={useTimer} onChange={(e) => setUseTimer(e.target.checked)} className="w-5 h-5 rounded" />
                  <label htmlFor="timer" className="text-slate-700 cursor-pointer">Enable 30 min Timer</label>
                </div>
                <button
                  onClick={startQuiz}
                  disabled={loading}
                  className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold shadow-md flex justify-center items-center gap-2"
                >
                  {loading ? <Loader2 className="animate-spin" /> : "Start Exam (40 Random)"}
                </button>
              </div>

              <button onClick={() => setShowAddForm(!showAddForm)} className="w-full py-3 border-2 border-dashed border-slate-300 text-slate-500 hover:border-slate-400 rounded-lg font-medium">
                + Contribute Question
              </button>
            </div>

            <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-slate-700"><History className="w-5 h-5" /> Recent Results</h2>
              {history.length === 0 ? <p className="text-slate-400 italic text-sm">No history yet.</p> : (
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {history.map((h, i) => (
                    <div key={i} className="flex justify-between text-sm bg-white p-2 rounded shadow-sm">
                      <span>{h.date}</span>
                      <span className={h.score / h.total >= 0.6 ? "text-green-600 font-bold" : "text-red-500 font-bold"}>{h.score}/{h.total}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Add Question Modal */}
          {showAddForm && (
            <div className="mt-8 pt-8 border-t animate-in fade-in">
              <h3 className="font-bold text-lg mb-4">Contribute to the Database</h3>
              <textarea placeholder="Question Text..." className="w-full p-3 border rounded mb-3" value={newQText} onChange={e => setNewQText(e.target.value)} />
              {newQOptions.map((opt, i) => (
                <div key={i} className="flex gap-2 mb-2">
                  <input type="radio" name="correct" checked={newQCorrect === i} onChange={() => setNewQCorrect(i)} className="mt-3" />
                  <input className="flex-1 p-2 border rounded" placeholder={`Option ${i + 1}`} value={opt} onChange={e => {
                    const n = [...newQOptions]; n[i] = e.target.value; setNewQOptions(n);
                  }} />
                </div>
              ))}
              <button onClick={() => setNewQOptions([...newQOptions, ""])} className="text-sm text-blue-600 mb-4">+ Option</button>
              <div className="flex justify-between items-center">
                <span className="text-green-600 font-medium">{submitStatus}</span>
                <div className="flex gap-2">
                  <button onClick={() => setShowAddForm(false)} className="px-4 py-2 text-slate-500">Cancel</button>
                  <button onClick={handleAddQuestion} className="px-4 py-2 bg-green-600 text-white rounded">Submit</button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  // --- VIEW: QUIZ & REVIEW ---
  const currentQ = activeQuestions[currentIndex];
  const isReview = quizState === 'review';

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col md:flex-row font-sans text-slate-800">
      {/* Sidebar */}
      <aside className="w-full md:w-20 lg:w-72 bg-white border-r flex flex-col h-auto md:h-screen sticky top-0 z-10">
        <div className="p-4 border-b flex justify-between items-center bg-slate-50">
          <h2 className="font-bold hidden lg:block">Questions</h2>
          {timeLeft !== null && !isReview && <div className={cn("flex gap-1 font-mono text-lg", timeLeft < 60 ? "text-red-600 animate-pulse" : "text-blue-600")}><Clock className="w-4 h-4" />{formatTime(timeLeft)}</div>}
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          <div className="grid grid-cols-5 md:grid-cols-1 lg:grid-cols-5 gap-2">
            {activeQuestions.map((q, idx) => {
              const status = isReview
                ? (userAnswers[q.id] === q.correct_answer ? 'correct' : 'wrong')
                : (currentIndex === idx ? 'current' : flagged.has(q.id) ? 'flagged' : userAnswers[q.id] !== undefined ? 'answered' : 'none');

              const colors = {
                current: "bg-blue-600 text-white scale-105 shadow-md",
                flagged: "bg-yellow-100 border-yellow-400 text-yellow-700",
                answered: "bg-blue-50 border-blue-200 text-blue-700",
                correct: "bg-green-100 border-green-300 text-green-700",
                wrong: "bg-red-100 border-red-300 text-red-700",
                none: "bg-white border-slate-200 text-slate-500 hover:bg-slate-50"
              };

              return (
                <button key={q.id} onClick={() => setCurrentIndex(idx)} className={cn("h-10 w-full rounded flex items-center justify-center text-sm font-medium border relative", colors[status])}>
                  {idx + 1}
                  {flagged.has(q.id) && !isReview && <Flag className="w-2.5 h-2.5 absolute top-0.5 right-0.5 fill-yellow-500 text-yellow-500" />}
                </button>
              );
            })}
          </div>
        </div>
        <div className="p-4 border-t bg-slate-50">
          <button onClick={isReview ? () => setQuizState('menu') : finishQuiz} className="w-full py-2 bg-slate-800 text-white rounded hover:bg-slate-900 transition flex items-center justify-center gap-2">
            {isReview ? <><RotateCcw className="w-4 h-4" /> Menu</> : "Submit Quiz"}
          </button>
        </div>
      </aside>

      {/* Main Card */}
      <main className="flex-1 p-4 md:p-8 lg:p-12 overflow-y-auto">
        <div className="max-w-3xl mx-auto bg-white rounded-2xl shadow-xl overflow-hidden min-h-[500px] flex flex-col">
          <div className="bg-slate-50 p-6 border-b flex justify-between items-start">
            <div>
              <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Question {currentIndex + 1}</span>
              <p className="mt-4 text-xl font-medium text-slate-800">{currentQ?.text}</p>
            </div>
            {!isReview && (
              <button onClick={() => setFlagged(prev => { const n = new Set(prev); n.has(currentQ.id) ? n.delete(currentQ.id) : n.add(currentQ.id); return n; })} className={cn("p-2 rounded-full", flagged.has(currentQ.id) ? "bg-yellow-100 text-yellow-600" : "text-slate-300 hover:bg-slate-100")}>
                <Flag className={cn("w-6 h-6", flagged.has(currentQ.id) && "fill-current")} />
              </button>
            )}
          </div>

          <div className="p-6 space-y-3 flex-1">
            {currentQ?.options.map((opt, i) => {
              const selected = userAnswers[currentQ.id] === i;
              const isCorrect = currentQ.correct_answer === i;

              let style = "border-slate-200 hover:bg-slate-50";
              if (isReview) {
                if (isCorrect) style = "border-green-500 bg-green-50 text-green-800";
                else if (selected) style = "border-red-300 bg-red-50 text-red-800";
                else style = "border-slate-100 text-slate-400";
              } else if (selected) {
                style = "border-blue-600 bg-blue-50 text-blue-800";
              }

              return (
                <button key={i} disabled={isReview} onClick={() => setUserAnswers(prev => ({ ...prev, [currentQ.id]: i }))}
                  className={cn("w-full text-left p-4 rounded-xl border-2 flex items-center gap-3 transition-all", style)}>
                  <div className={cn("w-6 h-6 rounded-full border-2 flex items-center justify-center text-[10px]", (selected || (isReview && isCorrect)) ? "border-current" : "border-slate-300")}>
                    {String.fromCharCode(65 + i)}
                  </div>
                  <span className="flex-1">{opt}</span>
                  {isReview && isCorrect && <CheckCircle2 className="w-5 h-5 text-green-600" />}
                  {isReview && selected && !isCorrect && <XCircle className="w-5 h-5 text-red-500" />}
                </button>
              );
            })}
          </div>

          <div className="p-6 border-t flex justify-between bg-slate-50">
            <button disabled={currentIndex === 0} onClick={() => setCurrentIndex(i => i - 1)} className="px-4 py-2 rounded-lg text-slate-600 hover:bg-slate-200 disabled:opacity-50 flex items-center gap-2"><ChevronLeft className="w-4 h-4" /> Prev</button>
            <button disabled={currentIndex === activeQuestions.length - 1} onClick={() => setCurrentIndex(i => i + 1)} className="px-6 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2">Next <ChevronRight className="w-4 h-4" /></button>
          </div>
        </div>
      </main>
    </div>
  );
}