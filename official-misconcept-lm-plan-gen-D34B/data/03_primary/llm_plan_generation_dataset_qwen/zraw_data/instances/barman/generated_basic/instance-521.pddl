(define (problem prob)
 (:domain barman)
 (:objects 
      shaker194 - shaker
      left right - hand
      shot137 shot116 - shot
      ingredient493 ingredient282 - ingredient
      cocktail1 - cocktail
      dispenser340 dispenser49 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker194)
  (ontable shot137)
  (ontable shot116)
  (dispenses dispenser340 ingredient493)
  (dispenses dispenser49 ingredient282)
  (clean shaker194)
  (clean shot137)
  (clean shot116)
  (empty shaker194)
  (empty shot137)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker194 l0)
  (shaker-level shaker194 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient493)
  (cocktail-part2 cocktail1 ingredient282)
)
 (:goal
  (and
      (contains shot137 cocktail1)
)))
