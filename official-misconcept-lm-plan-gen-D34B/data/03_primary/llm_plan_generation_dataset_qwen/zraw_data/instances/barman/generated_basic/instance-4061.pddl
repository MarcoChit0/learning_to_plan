(define (problem prob)
 (:domain barman)
 (:objects 
      shaker178 - shaker
      left right - hand
      shot189 shot280 - shot
      ingredient62 ingredient3 - ingredient
      cocktail482 - cocktail
      dispenser462 dispenser283 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker178)
  (ontable shot189)
  (ontable shot280)
  (dispenses dispenser462 ingredient62)
  (dispenses dispenser283 ingredient3)
  (clean shaker178)
  (clean shot189)
  (clean shot280)
  (empty shaker178)
  (empty shot189)
  (empty shot280)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker178 l0)
  (shaker-level shaker178 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail482 ingredient3)
  (cocktail-part2 cocktail482 ingredient62)
)
 (:goal
  (and
      (contains shot189 cocktail482)
)))
