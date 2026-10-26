import {
  BookOpen,
  Plus,
  MessageSquare,
  Upload,
  Layers,
  AlertCircle,
  CheckCircle2,
  X,
} from 'lucide-react';
import type { BookMetadata, Conversation, ChatMessage } from '../types/api';

interface SidebarProps {
  books: BookMetadata[];
  selectedBookId: string | null;
  onSelectBook: (bookId: string) => void;
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelectConversation: (convId: string) => void;
  onNewChat: () => void;
  onOpenUpload: () => void;
  onOpenHierarchy: () => void;
  isOpenMobile: boolean;
  onCloseMobile: () => void;
  isLoadingBooks?: boolean;
}

export function Sidebar({
  books,
  selectedBookId,
  onSelectBook,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewChat,
  onOpenUpload,
  onOpenHierarchy,
  isOpenMobile,
  onCloseMobile,
}: SidebarProps) {
  const selectedBook = books.find((b) => b.book_id === selectedBookId) || books[0];

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'indexed':
      case 'completed':
      case 'ready':
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-700 bg-emerald-50 border border-emerald-200/60 px-1.5 py-0.5 rounded">
            <CheckCircle2 className="w-2.5 h-2.5" />
            Ready
          </span>
        );
      case 'processing':
      case 'extracting':
      case 'indexing':
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-700 bg-amber-50 border border-amber-200/60 px-1.5 py-0.5 rounded">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
            Indexing
          </span>
        );
      case 'failed':
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-medium text-rose-700 bg-rose-50 border border-rose-200/60 px-1.5 py-0.5 rounded">
            <AlertCircle className="w-2.5 h-2.5" />
            Failed
          </span>
        );
      default:
        return (
          <span className="text-[11px] text-[#8C867C] bg-[#F2EDE4] px-1.5 py-0.5 rounded">
            {status}
          </span>
        );
    }
  };

  const content = (
    <div className="flex flex-col h-full bg-[#FAF9F5] border-r border-[#ECE7DE] text-[#2D2B28] select-none">
      {/* Mobile close button & Sidebar header */}
      <div className="p-4 border-b border-[#ECE7DE] flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-[#272522] flex items-center justify-center text-[#FAF9F5] shadow-xs">
            <BookOpen className="w-3.5 h-3.5 text-[#E6A07E]" />
          </div>
          <span className="font-serif-claude font-medium text-base text-[#22211F]">
            Library & Dialogues
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={onOpenUpload}
            title="Upload new book"
            className="p-1.5 rounded-md text-[#6B655B] hover:text-[#22211F] hover:bg-[#EFEAE1] transition-colors"
          >
            <Upload className="w-4 h-4" />
          </button>
          {isOpenMobile && (
            <button
              onClick={onCloseMobile}
              className="lg:hidden p-1.5 rounded-md text-[#6B655B] hover:text-[#22211F] hover:bg-[#EFEAE1]"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Book Selection Section */}
      <div className="p-3 border-b border-[#ECE7DE]">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold tracking-wider uppercase text-[#88837A]">
            Active Book
          </span>
          <span className="text-xs text-[#8C867C]">
            {books.length} {books.length === 1 ? 'book' : 'books'}
          </span>
        </div>

        {books.length > 0 ? (
          <div className="space-y-2">
            <select
              value={selectedBookId || ''}
              onChange={(e) => onSelectBook(e.target.value)}
              className="w-full text-xs rounded-lg border border-[#DED7CA] bg-[#FAF8F3] px-2.5 py-2 text-[#2D2B28] font-medium focus:border-[#C86432] focus:outline-none transition-colors"
            >
              {books.map((b) => (
                <option key={b.book_id} value={b.book_id}>
                  {b.title || b.original_filename} ({b.chunk_count || 0} chunks)
                </option>
              ))}
            </select>

            {selectedBook && (
              <div className="rounded-lg border border-[#ECE7DE] bg-[#F7F4EC]/60 p-2.5 text-xs space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-[#22211F] truncate max-w-[170px]" title={selectedBook.title || selectedBook.original_filename}>
                    {selectedBook.title || selectedBook.original_filename}
                  </span>
                  {getStatusBadge(selectedBook.indexing_status || selectedBook.extraction_status)}
                </div>
                <div className="flex items-center gap-2 text-[11px] text-[#787267]">
                  <span>{selectedBook.page_count} pages</span>
                  <span>•</span>
                  <span>{selectedBook.chunk_count || 0} chunks</span>
                </div>
                <button
                  onClick={onOpenHierarchy}
                  className="w-full mt-1.5 inline-flex items-center justify-center gap-1.5 rounded-md border border-[#DCD5C8] bg-[#FAF9F5] px-2 py-1 text-xs font-medium text-[#4D483F] hover:bg-[#F0ECE1] hover:text-[#22211F] transition-colors"
                >
                  <Layers className="w-3 h-3 text-[#B05C38]" />
                  <span>Crux & Hierarchy</span>
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="rounded-lg border border-dashed border-[#DDD7CC] p-4 text-center">
            <BookOpen className="w-6 h-6 text-[#A09A8F] mx-auto mb-1.5" />
            <p className="text-xs text-[#706B62] mb-2">No books uploaded yet</p>
            <button
              onClick={onOpenUpload}
              className="inline-flex items-center gap-1.5 text-xs font-medium text-[#C86432] hover:underline"
            >
              <Upload className="w-3.5 h-3.5" />
              Upload PDF
            </button>
          </div>
        )}
      </div>

      {/* New Chat action button */}
      <div className="p-3 border-b border-[#ECE7DE]">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 rounded-xl bg-[#24211E] px-3.5 py-2.5 text-xs font-medium text-[#FAF9F5] hover:bg-[#383430] active:scale-[0.99] transition-all shadow-xs"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>New Conversation</span>
        </button>
      </div>

      {/* Conversation Thread List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-1">
        <div className="flex items-center justify-between px-1 mb-1.5">
          <span className="text-[11px] font-semibold tracking-wider uppercase text-[#88837A]">
            Conversations
          </span>
          <span className="text-[11px] text-[#8C867C] font-mono-code">
            {conversations.length}
          </span>
        </div>

        {conversations.length > 0 ? (
          conversations.map((conv) => {
            const isActive = conv.conversation_id === activeConversationId;
            const firstUserMsg = conv.messages.find((m: ChatMessage) => m.role === 'user')?.content || 'New inquiry';
            const messageCount = conv.messages.length;

            return (
              <button
                key={conv.conversation_id}
                onClick={() => {
                  onSelectConversation(conv.conversation_id);
                  if (isOpenMobile) onCloseMobile();
                }}
                className={`w-full text-left rounded-lg p-2.5 text-xs transition-colors flex items-start gap-2.5 group ${
                  isActive
                    ? 'bg-[#EAE4D7] text-[#22211F] font-medium'
                    : 'text-[#5C564D] hover:bg-[#F4EFE6] hover:text-[#22211F]'
                }`}
              >
                <MessageSquare className={`w-3.5 h-3.5 mt-0.5 flex-shrink-0 ${
                  isActive ? 'text-[#C86432]' : 'text-[#8C867C] group-hover:text-[#4A4741]'
                }`} />
                <div className="flex-1 min-w-0">
                  <p className="truncate text-xs leading-snug">
                    {firstUserMsg}
                  </p>
                  <div className="flex items-center gap-1.5 text-[10px] text-[#8C867C] mt-1 font-sans">
                    <span>{messageCount} {messageCount === 1 ? 'msg' : 'msgs'}</span>
                    <span>•</span>
                    <span className="truncate">
                      {new Date(conv.updated_at || conv.created_at).toLocaleDateString([], {
                        month: 'short',
                        day: 'numeric',
                      })}
                    </span>
                  </div>
                </div>
              </button>
            );
          })
        ) : (
          <div className="py-8 text-center text-xs text-[#8C867C]">
            <p>No conversations yet</p>
            <p className="text-[11px] text-[#A8A296] mt-1">Start by asking a question about the active book</p>
          </div>
        )}
      </div>

      {/* Bottom info footer */}
      <div className="p-3 border-t border-[#ECE7DE] bg-[#FAF8F3] text-[11px] text-[#8C867C] flex items-center justify-between">
        <span className="font-mono-code">BookMind v1.0</span>
        <button
          onClick={onOpenUpload}
          className="text-[#C86432] hover:underline font-medium"
        >
          + Add Book
        </button>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop static sidebar */}
      <aside className="hidden lg:block w-72 flex-shrink-0 h-screen sticky top-0 overflow-hidden">
        {content}
      </aside>

      {/* Mobile drawer overlay */}
      {isOpenMobile && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="fixed inset-0 bg-black/30 backdrop-blur-xs transition-opacity"
            onClick={onCloseMobile}
          />
          <div className="fixed inset-y-0 left-0 w-80 max-w-[85vw] shadow-xl z-50">
            {content}
          </div>
        </div>
      )}
    </>
  );
}
