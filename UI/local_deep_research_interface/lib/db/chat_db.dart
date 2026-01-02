import 'package:path/path.dart';
import 'package:sqflite/sqflite.dart';

class LocalDb {
  static Database? _db;

  static Future<Database> get db async {
    if (_db != null) return _db!;
    _db = await _init();
    return _db!;
  }

  static Future<Database> _init() async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, 'chat_app.db');

    return openDatabase(
      path,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE sessions(
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at INTEGER NOT NULL
          )
        ''');

        await db.execute('''
          CREATE TABLE messages(
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            text TEXT NOT NULL,
            is_user INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            response_list TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
          )
        ''');

        await db.execute('CREATE INDEX idx_messages_session ON messages(session_id)');
      },
    );
  }

  // ---------- Sessions ----------
  static Future<void> upsertSession({
    required String id,
    required String title,
    required DateTime createdAt,
  }) async {
    final database = await db;
    await database.insert(
      'sessions',
      {
        'id': id,
        'title': title,
        'created_at': createdAt.millisecondsSinceEpoch,
      },
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  static Future<List<Map<String, Object?>>> getAllSessions() async {
    final database = await db;
    return database.query(
      'sessions',
      orderBy: 'created_at DESC',
    );
  }

  static Future<void> deleteSession(String sessionId) async {
    final database = await db;
    await database.delete('messages', where: 'session_id = ?', whereArgs: [sessionId]);
    await database.delete('sessions', where: 'id = ?', whereArgs: [sessionId]);
  }

  // ---------- Messages ----------
  static Future<void> insertMessage({
    required String id,
    required String sessionId,
    required String text,
    required bool isUser,
    required DateTime timestamp,
    required String responseListJson,
  }) async {
    final database = await db;
    await database.insert(
      'messages',
      {
        'id': id,
        'session_id': sessionId,
        'text': text,
        'is_user': isUser ? 1 : 0,
        'timestamp': timestamp.millisecondsSinceEpoch,
        'response_list': responseListJson,
      },
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  static Future<List<Map<String, Object?>>> getMessagesOfSession(String sessionId) async {
    final database = await db;
    return database.query(
      'messages',
      where: 'session_id = ?',
      whereArgs: [sessionId],
      orderBy: 'timestamp ASC',
    );
  }
}
