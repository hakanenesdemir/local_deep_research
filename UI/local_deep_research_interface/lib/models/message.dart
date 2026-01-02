import 'dart:convert';

class Message {
  final String id;
  final String text;
  final bool isUser;
  final DateTime timestamp;
  final List responseList;

  Message(
    this.responseList, {
    required this.id,
    required this.text,
    required this.isUser,
    required this.timestamp,
  });

  String responseListToJson() => jsonEncode(responseList);

  static List responseListFromJson(String s) {
    try { return jsonDecode(s) as List; } catch (_) { return []; }
  }
}
