(define (problem prob)
 (:domain barman)
 (:objects 
      shaker487 - shaker
      left right - hand
      shot12 shot399 - shot
      ingredient4 ingredient262 ingredient48 - ingredient
      cocktail1 - cocktail
      dispenser142 dispenser217 dispenser459 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker487)
  (ontable shot12)
  (ontable shot399)
  (dispenses dispenser142 ingredient4)
  (dispenses dispenser217 ingredient262)
  (dispenses dispenser459 ingredient48)
  (clean shaker487)
  (clean shot12)
  (clean shot399)
  (empty shaker487)
  (empty shot12)
  (empty shot399)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker487 l0)
  (shaker-level shaker487 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient48)
  (cocktail-part2 cocktail1 ingredient262)
)
 (:goal
  (and
      (contains shot12 cocktail1)
)))
