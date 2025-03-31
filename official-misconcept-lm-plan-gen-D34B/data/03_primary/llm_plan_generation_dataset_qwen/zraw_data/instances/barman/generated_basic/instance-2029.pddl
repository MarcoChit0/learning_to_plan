(define (problem prob)
 (:domain barman)
 (:objects 
      shaker379 - shaker
      left right - hand
      shot106 shot375 - shot
      ingredient282 ingredient326 - ingredient
      cocktail403 - cocktail
      dispenser184 dispenser130 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker379)
  (ontable shot106)
  (ontable shot375)
  (dispenses dispenser184 ingredient282)
  (dispenses dispenser130 ingredient326)
  (clean shaker379)
  (clean shot106)
  (clean shot375)
  (empty shaker379)
  (empty shot106)
  (empty shot375)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker379 l0)
  (shaker-level shaker379 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail403 ingredient282)
  (cocktail-part2 cocktail403 ingredient326)
)
 (:goal
  (and
      (contains shot106 cocktail403)
)))
