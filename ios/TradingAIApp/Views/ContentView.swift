import SwiftUI
import CoreData

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.managedObjectContext) private var ctx
    @StateObject private var portfoliosVM: PortfolioListViewModel
    @ObservedObject private var quotes = QuoteService.shared

    init() {
        let context = PersistenceController.shared.container.viewContext
        _portfoliosVM = StateObject(wrappedValue: PortfolioListViewModel(context: context))
    }

    var body: some View {
        TabView {
            NavigationView { portfolioList } .tabItem { Label("Portfolios", systemImage: "briefcase") }
            NavigationView { quotesView } .tabItem { Label("Quotes", systemImage: "chart.line.uptrend.xyaxis") }
            NavigationView { settingsView } .tabItem { Label("Settings", systemImage: "gearshape") }
        }
    }

    private var portfolioList: some View {
        List {
            ForEach(portfoliosVM.portfolios) { p in
                VStack(alignment: .leading) {
                    Text(p.name).font(.headline)
                    Text(p.createdAt, style: .date).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .navigationTitle("Portfolios")
        .toolbar {
            Button(action: portfoliosVM.addSample) { Image(systemName: "plus") }
        }
    }

    private var quotesView: some View {
        List(quotes.quotes.values.sorted { $0.symbol < $1.symbol }) { q in
            HStack {
                Text(q.symbol).bold()
                Spacer()
                VStack(alignment: .trailing) {
                    Text(String(format: "%.2f", q.last))
                    Text(String(format: "%+.2f%%", q.changePct)).font(.caption)
                        .foregroundColor(q.changePct >= 0 ? .green : .red)
                }
            }
        }
        .navigationTitle("Quotes")
    }

    private var settingsView: some View {
        Form {
            Toggle("iCloud Sync", isOn: $appState.isCloudSyncEnabled)
            Toggle("Notifications", isOn: $appState.notificationsEnabled)
        }
        .navigationTitle("Settings")
    }
}

#Preview {
    ContentView()
        .environmentObject(AppState())
        .environment(\.managedObjectContext, PersistenceController.shared.container.viewContext)
}
