(define (problem prob)
 (:domain barman)
 (:objects 
      shaker116 - shaker
      left right - hand
      shot321 shot399 shot217 - shot
      ingredient26 ingredient309 - ingredient
      cocktail189 - cocktail
      dispenser244 dispenser441 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker116)
  (ontable shot321)
  (ontable shot399)
  (ontable shot217)
  (dispenses dispenser244 ingredient26)
  (dispenses dispenser441 ingredient309)
  (clean shaker116)
  (clean shot321)
  (clean shot399)
  (clean shot217)
  (empty shaker116)
  (empty shot321)
  (empty shot399)
  (empty shot217)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker116 l0)
  (shaker-level shaker116 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail189 ingredient309)
  (cocktail-part2 cocktail189 ingredient26)
)
 (:goal
  (and
      (contains shot321 cocktail189)
      (contains shot399 cocktail189)
)))
