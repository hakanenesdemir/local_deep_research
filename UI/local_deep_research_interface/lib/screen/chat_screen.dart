
import 'package:flutter/material.dart';
import 'package:local_deep_research_interface/models/chat_session.dart';
import 'package:local_deep_research_interface/models/message.dart';
import 'package:local_deep_research_interface/services/api_services.dart';
import 'package:local_deep_research_interface/widgets/chat_bubble.dart';
import 'package:local_deep_research_interface/widgets/left_drawer.dart';
import 'package:local_deep_research_interface/widgets/welcome.dart';
import 'package:uuid/uuid.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  // State Değişkenleri
  final Uuid _uuid = const Uuid();
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  // Veri Durumu
  final List<ChatSession> _chatHistory = []; 
  String? _currentSessionId; 
  bool _isWaitingForResponse = false; 

  // Şu anki mesajları getiren getter
  List<Message> get _currentMessages {
    if (_currentSessionId == null) return [];
    try {
      return _chatHistory
          .firstWhere((element) => element.id == _currentSessionId)
          .messages;
    } catch (e) {
      return [];
    }
  }

  // Şu anki oturumun başlığı
  String get _currentTitle {
    if (_currentSessionId == null) return "Yeni Sohbet";
    try {
      return _chatHistory
          .firstWhere((e) => e.id == _currentSessionId)
          .title;
    } catch (e) {
      return "Hata";
    }
  }

  // --- MANTIK FONKSİYONLARI ---

  void _createNewPage() {
    setState(() {
      _currentSessionId = null;
      _isWaitingForResponse = false;
    });
  }

  void _loadSession(String sessionId) {
    setState(() {
      _currentSessionId = sessionId;
      _isWaitingForResponse = false;
    });
    _scrollToBottom();
  }

  void _deleteSession(String sessionId) {
    setState(() {
      _chatHistory.removeWhere((s) => s.id == sessionId);
      if (_currentSessionId == sessionId) {
        _currentSessionId = null;
      }
    });
  }

  // MESAJ GÖNDERME (SENİN MANTIĞIN ENTEGRE EDİLDİ)
  Future<void> _handleSendMessage() async {
    final text = _textController.text.trim();
    if (text.isEmpty) return;

    _textController.clear();

    // 1. Kullanıcı mesajını oluştur ve ekrana ekle
    final userMessage = Message(
      id: _uuid.v4(),
      text: text,
      isUser: true,
      timestamp: DateTime.now(),
      []
    );

    setState(() {
      if (_currentSessionId == null) {
        // Yeni oturum
        final newId = _uuid.v4();
        final newSession = ChatSession(
          id: newId,
          title: text.length > 20 ? "${text.substring(0, 20)}..." : text,
          messages: [userMessage],
          createdAt: DateTime.now(),
        );
        _chatHistory.insert(0, newSession);
        _currentSessionId = newId;
      } else {
        // Mevcut oturuma ekle
        final session = _chatHistory.firstWhere((s) => s.id == _currentSessionId);
        session.messages.add(userMessage);
      }
      
      _isWaitingForResponse = true; // Yükleniyor...
    });

    _scrollToBottom();

    try {
      // 2. API'ye İsteği Gönder (ApiService içindeki http isteği)
      final responseText = await ApiService.getApiResponse(text);
      debugPrint("responseText.toString() $responseText.toString()");

      if (!mounted) return;

      // 3. Gelen Cevabı Ekrana Ekle
      final botMessage = Message(
        id: _uuid.v4(),
        text: responseText[0],
        isUser: false,
        timestamp: DateTime.now(),
        responseText[2],
      );

      setState(() {
        final session = _chatHistory.firstWhere((s) => s.id == _currentSessionId);
        session.messages.add(botMessage);
        _isWaitingForResponse = false; // Yükleme bitti
      });
      
      _scrollToBottom();

    } catch (e) {
      // Hata yakalama (ApiService zaten string dönüyor ama yine de ekstra güvenlik)
      setState(() {
         _isWaitingForResponse = false;
         final session = _chatHistory.firstWhere((s) => s.id == _currentSessionId);
         session.messages.add(
          Message(
           id: _uuid.v4(),
           text: "Beklenmeyen bir hata: $e",
           isUser: false,
           timestamp: DateTime.now(),
           []
         ));
      });
    }
  }

  // Özet İste
  Future<void> _handleSummarize() async {
    if (_currentSessionId == null) return;

    setState(() {
      _isWaitingForResponse = true;
    });
    
    _scrollToBottom();

    try {
      final currentMsgs = _currentMessages;
      final lastMessage = currentMsgs.last;
      final lastMessageText = lastMessage.text;
      final summaryText = await ApiService.getSummary(lastMessageText);

      if (!mounted) return;

      final summaryMessage = Message(
        id: _uuid.v4(),
        text: summaryText,
        isUser: false,
        timestamp: DateTime.now(),
        []
      );

      setState(() {
        final session = _chatHistory.firstWhere((s) => s.id == _currentSessionId);
        session.messages.add(summaryMessage);
        _isWaitingForResponse = false;
      });

      _scrollToBottom();

    } catch (e) {
      setState(() => _isWaitingForResponse = false);
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  // ==========================================
  // ARAYÜZ (UI)
  // ==========================================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_currentTitle, style: const TextStyle(fontSize: 16)),
        actions: [
          if (_currentSessionId != null && !_isWaitingForResponse)
            IconButton(
              icon: const Icon(Icons.summarize),
              tooltip: "Özetle",
              onPressed: _handleSummarize,
            ),
        ],
      ),
      
      drawer: ChatDrawer(
        chatHistory: _chatHistory,
        currentSessionId: _currentSessionId,
        onNewChat: _createNewPage,
        onLoadSession: _loadSession,
        onDeleteSession: _deleteSession,
      ),

      body: Column(
        children: [
          Expanded(
            child: _currentSessionId == null || (_currentMessages.isEmpty && !_isWaitingForResponse)
                ? WelcomeWidget()
                : ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.all(16),
                    itemCount: _currentMessages.length + (_isWaitingForResponse ? 1 : 0),
                    itemBuilder: (context, index) {
                      if (index == _currentMessages.length) {
                        return const Center(
                          child: Padding(
                            padding: EdgeInsets.all(16.0),
                            child: CircularProgressIndicator(),
                          ),
                        );
                      }
                      final msg = _currentMessages[index];
                      return MessageBubble(message: msg);
                    },
                  ),
          ),
          Container(
            padding: const EdgeInsets.all(8.0),
            decoration: BoxDecoration(
              color: Colors.white,
              boxShadow: [BoxShadow(color: Colors.black12, blurRadius: 4)],
            ),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _textController,
                    enabled: !_isWaitingForResponse,
                    decoration: const InputDecoration(
                      hintText: "Bir mesaj yazın...",
                      border: InputBorder.none,
                      contentPadding: EdgeInsets.symmetric(horizontal: 16),
                    ),
                    onSubmitted: (_) => _handleSendMessage(),
                  ),
                ),
                IconButton(
                  icon: Icon(Icons.send, 
                    color: _isWaitingForResponse ? Colors.grey : Colors.indigo
                  ),
                  onPressed: _isWaitingForResponse ? null : _handleSendMessage,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }


}
