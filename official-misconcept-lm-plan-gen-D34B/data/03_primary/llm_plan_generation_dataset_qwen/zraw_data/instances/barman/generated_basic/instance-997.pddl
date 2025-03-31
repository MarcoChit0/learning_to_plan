(define (problem prob)
 (:domain barman)
 (:objects 
      shaker119 - shaker
      left right - hand
      shot51 shot178 - shot
      ingredient241 ingredient182 - ingredient
      cocktail1 - cocktail
      dispenser473 dispenser294 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker119)
  (ontable shot51)
  (ontable shot178)
  (dispenses dispenser473 ingredient241)
  (dispenses dispenser294 ingredient182)
  (clean shaker119)
  (clean shot51)
  (clean shot178)
  (empty shaker119)
  (empty shot51)
  (empty shot178)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker119 l0)
  (shaker-level shaker119 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient241)
  (cocktail-part2 cocktail1 ingredient182)
)
 (:goal
  (and
      (contains shot51 cocktail1)
)))
