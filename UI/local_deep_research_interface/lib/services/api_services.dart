import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class ApiService {
  static Future<List> getApiResponse(String message) async {
    String baseUrl = Platform.isAndroid
        ? 'http://10.0.2.2:8000'
        : 'http://127.0.0.1:8000';
    final url = Uri.parse('$baseUrl/ask/question/ai');
    final Map<String, dynamic> body = {'girilen_metin': message};

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(body),
      );

      if (response.statusCode == 200) {
        final decodedBody = utf8.decode(response.bodyBytes);
        final jsonResponse = jsonDecode(decodedBody);

        // --- DÜZELTME BURADA ---
        if (jsonResponse['aiResponse'] != null) {
          // Gelen veri bir Liste (List<dynamic>)
          List<dynamic> sonuclar = jsonResponse['aiResponse'];
          debugPrint("sonuclar = $sonuclar");

          if (sonuclar.isEmpty) {
            return ["Veritabanında ilgili sonuç bulunamadı.", false];
          }

          // Listeden sadece metinleri alıp birleştirelim
          // Örnek: "1. Metin parçası... \n\n 2. Metin parçası..."
          StringBuffer buffer = StringBuffer();
          buffer.writeln("Bulunan Sonuçlar:\n");

          for (var item in sonuclar) {
            buffer.writeln(
              "📄 Kaynak: ${item['dosya_adi']} (Skor: ${item['benzerlik']})",
            );
            buffer.writeln("${item['metin']}");
            buffer.writeln("\n-------------------\n");
          }

          return [buffer.toString(), true, sonuclar];
        } else {
          return ["Sunucudan boş veri döndü (aiResponse null).", false];
        }
        // -----------------------
      } else {
        return ["Sunucu Hatası: ${response.statusCode}", false];
      }
    } catch (e) {
      return ["Bağlantı Hatası: $e", false];
    }
  }

  // Özetleme için ayrı bir endpointin varsa burayı da http ile güncelleyebilirsin.
  // Şimdilik aynı fonksiyonu çağırarak simüle ediyoruz veya prompt değiştirerek atabiliriz.
  // Bütün konuşmayı özetle
  static Future<String> getSummary(String girilen_metin) async {
    String baseUrl = Platform.isAndroid
        ? 'http://10.0.2.2:8000'
        : 'http://127.0.0.1:8000';

    try {
      final response = await http.post(
        Uri.parse("$baseUrl/summary"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"girilen_metin": girilen_metin}),
      );

      if (response.statusCode == 200) {
        final jsonResponse = jsonDecode(response.body);

        if (jsonResponse["status"] == "success") {
          return jsonResponse["summary"];
        } else {
          return "Hata: ${jsonResponse['message']}";
        }
      } else {
        return "HTTP Error: ${response.statusCode}";
      }
    } catch (e) {
      return "Bağlantı hatası: $e";
    }
  }
}
