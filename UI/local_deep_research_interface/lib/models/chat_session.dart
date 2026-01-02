import 'package:local_deep_research_interface/models/message.dart';

class ChatSession {
  final String id;
  String title;
  final List<Message> messages;
  final DateTime createdAt;

  ChatSession({
    required this.id,
    required this.title,
    required this.messages,
    required this.createdAt,
  });
}