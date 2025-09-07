import Foundation
import CoreData
import Combine

final class PortfolioListViewModel: ObservableObject {
    @Published var portfolios: [Portfolio] = []
    private var ctx: NSManagedObjectContext

    init(context: NSManagedObjectContext) {
        self.ctx = context
        load()
    }

    func load() {
        let req: NSFetchRequest<Portfolio> = Portfolio.fetchRequest()
        req.sortDescriptors = [NSSortDescriptor(key: "createdAt", ascending: true)]
        do {
            portfolios = try ctx.fetch(req)
        } catch {
            print("Fetch error: \(error)")
        }
    }

    func addSample() {
        let p = Portfolio(context: ctx)
        p.id = UUID()
        p.name = "Core"
        p.createdAt = Date()
        save()
    }

    func save() {
        do { try ctx.save(); load() } catch { print("Save error: \(error)") }
    }
}
