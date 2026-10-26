import { useState, useEffect, useCallback } from 'react';
import { Header, type NavTab } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ChatWorkspace } from './components/ChatWorkspace';
import { EvaluationDashboard } from './pages/EvaluationDashboard';
import { Home } from './pages/Home';
import { UploadModal } from './components/UploadModal';
import { HierarchyModal } from './components/HierarchyModal';
import {
  checkBackendHealth,
  listBooks,
  listConversations,
  getConversation,
  getHierarchy,
  chatWithBook,
} from './lib/api';
import type {
  BookMetadata,
  Conversation,
  ChatMessage,
  BookHierarchy,
  HealthState,
  PipelineType,
} from './types/api';

export function App() {
  const [activeTab, setActiveTab] = useState<NavTab>('chat');
  const [books, setBooks] = useState<BookMetadata[]>([]);
  const [selectedBookId, setSelectedBookId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [hierarchy, setHierarchy] = useState<BookHierarchy | null>(null);

  const [isLoadingBooks, setIsLoadingBooks] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [lastUserPrompt, setLastUserPrompt] = useState<{
    text: string;
    options?: { pipeline?: PipelineType; enableRerank?: boolean; topK?: number };
  } | null>(null);

  // Modals & Navigation state
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [hierarchyModalOpen, setHierarchyModalOpen] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  // Health state
  const [healthState, setHealthState] = useState<HealthState>({
    data: null,
    loading: true,
    error: null,
    latencyMs: null,
    lastChecked: null,
  });

  const performHealthCheck = useCallback(async () => {
    setHealthState((prev) => ({ ...prev, loading: true }));
    const result = await checkBackendHealth();
    setHealthState({
      data: result.data,
      loading: false,
      error: result.error,
      latencyMs: result.latencyMs,
      lastChecked: new Date(),
    });
  }, []);

  // Fetch books on load
  const refreshBooks = useCallback(async () => {
    setIsLoadingBooks(true);
    try {
      const bookList = await listBooks();
      setBooks(bookList || []);
      if (bookList && bookList.length > 0) {
        setSelectedBookId((prev) => (prev ? prev : bookList[0].book_id));
      }
    } catch (err) {
      console.error('Failed to list books:', err);
    } finally {
      setIsLoadingBooks(false);
    }
  }, []);

  useEffect(() => {
    performHealthCheck();
    refreshBooks();
  }, [performHealthCheck, refreshBooks]);

  // When selected book changes, fetch conversations and hierarchy
  useEffect(() => {
    if (!selectedBookId) {
      setConversations([]);
      setMessages([]);
      setHierarchy(null);
      return;
    }

    let isCurrent = true;

    async function loadBookData() {
      try {
        const [convList, hier] = await Promise.all([
          listConversations(selectedBookId!),
          getHierarchy(selectedBookId!),
        ]);

        if (!isCurrent) return;
        setConversations(convList);
        setHierarchy(hier);

        if (convList.length > 0) {
          setActiveConversationId(convList[0].conversation_id);
          setMessages(convList[0].messages || []);
        } else {
          setActiveConversationId(null);
          setMessages([]);
        }
      } catch (err) {
        console.error('Error loading book data:', err);
      }
    }

    loadBookData();
    return () => {
      isCurrent = false;
    };
  }, [selectedBookId]);

  // Handle selecting a specific conversation
  const handleSelectConversation = useCallback(
    async (convId: string) => {
      setActiveConversationId(convId);
      const found = conversations.find((c) => c.conversation_id === convId);
      if (found) {
        setMessages(found.messages || []);
      } else if (selectedBookId) {
        try {
          const detail = await getConversation(selectedBookId, convId);
          setMessages(detail.messages || []);
        } catch (e) {
          console.error('Failed to fetch conversation details:', e);
        }
      }
    },
    [conversations, selectedBookId]
  );

  // Handle starting a new conversation thread
  const handleNewChat = useCallback(() => {
    setActiveConversationId(null);
    setMessages([]);
    setChatError(null);
  }, []);

  // Handle sending a message in chat
  const handleSendMessage = async (
    text: string,
    options?: {
      pipeline?: PipelineType;
      enableRerank?: boolean;
      topK?: number;
    }
  ) => {
    if (!selectedBookId) return;

    setLastUserPrompt({ text, options });
    setChatError(null);

    // Optimistically add user message
    const userMsg: ChatMessage = {
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setChatLoading(true);

    try {
      const response = await chatWithBook(selectedBookId, {
        message: text,
        conversation_id: activeConversationId || undefined,
        top_k: options?.topK ?? 5,
        enable_reranker: options?.enableRerank ?? false,
        include_hierarchy: options?.pipeline !== 'baseline_rag',
      });

      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: response.answer,
        citations: response.citations,
        question_type: response.question_type,
        insufficient_evidence: response.insufficient_evidence,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMsg]);
      setActiveConversationId(response.conversation_id);

      // Refresh conversations list to update order/previews
      const updatedConvs = await listConversations(selectedBookId);
      setConversations(updatedConvs);
    } catch (err: any) {
      setChatError(err.message || 'An error occurred while communicating with the assistant.');
    } finally {
      setChatLoading(false);
    }
  };

  const handleRetryLastMessage = () => {
    if (lastUserPrompt) {
      // Remove last user message before retrying to avoid duplicates
      setMessages((prev) => prev.slice(0, prev.length - 1));
      handleSendMessage(lastUserPrompt.text, lastUserPrompt.options);
    }
  };

  const selectedBook = books.find((b) => b.book_id === selectedBookId) || null;

  return (
    <div className="min-h-screen bg-[#FAF9F5] text-[#2D2B28] font-sans-ui flex flex-col selection:bg-[#E8DCC4] selection:text-[#1F1E1D]">
      {/* Header bar with navigation and live status */}
      <Header
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        healthState={healthState}
        onRefresh={performHealthCheck}
        onOpenMobileSidebar={() => setMobileSidebarOpen(true)}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {activeTab === 'chat' && (
          <div className="flex-1 flex w-full h-[calc(100vh-60px)] overflow-hidden">
            <Sidebar
              books={books}
              selectedBookId={selectedBookId}
              onSelectBook={setSelectedBookId}
              conversations={conversations}
              activeConversationId={activeConversationId}
              onSelectConversation={handleSelectConversation}
              onNewChat={handleNewChat}
              onOpenUpload={() => setUploadModalOpen(true)}
              onOpenHierarchy={() => setHierarchyModalOpen(true)}
              isOpenMobile={mobileSidebarOpen}
              onCloseMobile={() => setMobileSidebarOpen(false)}
              isLoadingBooks={isLoadingBooks}
            />

            <ChatWorkspace
              book={selectedBook}
              hierarchy={hierarchy}
              messages={messages}
              isLoading={chatLoading}
              error={chatError}
              onSendMessage={handleSendMessage}
              onRetryLastMessage={handleRetryLastMessage}
              onOpenHierarchy={() => setHierarchyModalOpen(true)}
              onOpenUpload={() => setUploadModalOpen(true)}
            />
          </div>
        )}

        {activeTab === 'evaluation' && (
          <div className="flex-1 overflow-y-auto">
            <EvaluationDashboard
              books={books}
              activeBookId={selectedBookId}
              onSelectBook={setSelectedBookId}
            />
          </div>
        )}

        {activeTab === 'health' && (
          <div className="flex-1 overflow-y-auto">
            <Home healthState={healthState} onRefreshHealth={performHealthCheck} />
          </div>
        )}
      </div>

      {/* Ingestion & PDF Upload Modal */}
      <UploadModal
        isOpen={uploadModalOpen}
        onClose={() => setUploadModalOpen(false)}
        onUploadSuccess={(newBook: BookMetadata) => {
          refreshBooks();
          setSelectedBookId(newBook.book_id);
          setActiveTab('chat');
        }}
      />

      {/* Hierarchical Knowledge (Levels 2-4) Modal */}
      {selectedBook && (
        <HierarchyModal
          isOpen={hierarchyModalOpen}
          onClose={() => setHierarchyModalOpen(false)}
          book={selectedBook}
        />
      )}
    </div>
  );
}

export default App;
