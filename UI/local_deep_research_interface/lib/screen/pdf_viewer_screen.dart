// PDF'in görüneceği ayrı sayfa (Artık Dinamik!)
import 'package:flutter/material.dart';
import 'package:pdfrx/pdfrx.dart';

/* onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const PdfGoruntulemeSayfasi(
                      pdfDosyaYolu: 'assets/kullanim_kilavuzu.pdf', 
                      baslik: 'Kılavuz',
                    ),
                  ),
                );
              }, */

class PdfViewerScreen extends StatelessWidget {
  final String pdfDosyaYolu;

  const PdfViewerScreen({
    super.key,
    required this.pdfDosyaYolu, // Bu alan zorunlu
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(automaticallyImplyLeading: true),
      // pdfrx kullanımı:
      body: PdfViewer.asset(
        "assets/pdfs/$pdfDosyaYolu", // Artık hardcoded değil, değişkenden geliyor
        params: PdfViewerParams(
          minScale: 1.0,
          maxScale: 3.0,
          errorBannerBuilder: (context, error, stackTrace, documentRef) {
            return Center(child: Text("Hata oluştu: $error"));
          },
        ),
      ),
    );
  }
}
