(define (problem prob)
 (:domain barman)
 (:objects 
      shaker308 - shaker
      left right - hand
      shot34 shot315 - shot
      ingredient232 ingredient446 ingredient303 - ingredient
      cocktail123 - cocktail
      dispenser202 dispenser120 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker308)
  (ontable shot34)
  (ontable shot315)
  (dispenses dispenser202 ingredient232)
  (dispenses dispenser120 ingredient446)
  (dispenses dispenser214 ingredient303)
  (clean shaker308)
  (clean shot34)
  (clean shot315)
  (empty shaker308)
  (empty shot34)
  (empty shot315)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker308 l0)
  (shaker-level shaker308 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail123 ingredient303)
  (cocktail-part2 cocktail123 ingredient446)
)
 (:goal
  (and
      (contains shot34 cocktail123)
)))
