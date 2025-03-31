(define (problem prob)
 (:domain barman)
 (:objects 
      shaker16 - shaker
      left right - hand
      shot68 shot120 - shot
      ingredient309 ingredient303 ingredient195 - ingredient
      cocktail1 - cocktail
      dispenser372 dispenser17 dispenser137 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker16)
  (ontable shot68)
  (ontable shot120)
  (dispenses dispenser372 ingredient309)
  (dispenses dispenser17 ingredient303)
  (dispenses dispenser137 ingredient195)
  (clean shaker16)
  (clean shot68)
  (clean shot120)
  (empty shaker16)
  (empty shot68)
  (empty shot120)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker16 l0)
  (shaker-level shaker16 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient309)
  (cocktail-part2 cocktail1 ingredient195)
)
 (:goal
  (and
      (contains shot68 cocktail1)
)))
