import SwiftUI
import CoreData

@main
struct TradingAIApp: App {
    @StateObject private var appState = AppState()
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}

final class AppState: ObservableObject {
    @Published var isCloudSyncEnabled: Bool = true
    @Published var notificationsEnabled: Bool = true
    @Published var selectedPortfolioId: UUID? = nil
}
