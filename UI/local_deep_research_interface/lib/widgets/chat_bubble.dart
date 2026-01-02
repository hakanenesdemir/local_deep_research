import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:local_deep_research_interface/models/message.dart';
import 'package:local_deep_research_interface/screen/pdf_viewer_screen.dart';

class MessageBubble extends StatelessWidget {
  final Message message;
  const MessageBubble({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    final isUser = message.isUser;
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75,
        ),
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: isUser ? Colors.indigo : Colors.grey.shade200,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              message.text,
              style: TextStyle(
                color: isUser ? Colors.white : Colors.black87,
                fontSize: 16,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              DateFormat('HH:mm').format(message.timestamp),
              style: TextStyle(
                color: isUser ? Colors.white70 : Colors.black54,
                fontSize: 10,
              ),
            ),
            isUser
                ? SizedBox()
                : Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: List.generate(message.responseList.length, (
                      index,
                    ) {
                      // index 0'dan başlar: 0, 1, 2, 3, 4
                      return Column(
                        children: [
                          Text("${index + 1}. pdf"),
                          IconButton(
                            onPressed: () {
                              String dosyaYolu =
                                  message.responseList[index]["dosya_adi"];
                              debugPrint("dosyayoku = $dosyaYolu");
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) =>
                                      PdfViewerScreen(pdfDosyaYolu: dosyaYolu),
                                ),
                              );
                            },
                            icon: Icon(Icons.folder_open),
                          ),
                        ],
                      );
                    }),
                  ),
          ],
        ),
      ),
    );
  }
}
