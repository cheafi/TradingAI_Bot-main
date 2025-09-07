import Foundation
import Combine

final class QuoteService: ObservableObject {
    static let shared = QuoteService()

    @Published private(set) var quotes: [String: Quote] = [:]
    private var timer: AnyCancellable?

    private init() {
        start()
    }

    func start() {
        timer = Timer.publish(every: 5, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                self?.mockUpdate()
            }
    }

    private func mockUpdate() {
        // Simulate price changes
        let symbols = ["AAPL", "MSFT", "TSLA", "BTC-USD"]
        for s in symbols {
            let price = Double.random(in: 100...500)
            quotes[s] = Quote(symbol: s, last: price, changePct: Double.random(in: -2...2))
        }
    }
}

struct Quote: Identifiable {
    var id: String { symbol }
    let symbol: String
    let last: Double
    let changePct: Double
}
