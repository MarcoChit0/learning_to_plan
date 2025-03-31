(define (problem prob)
 (:domain barman)
 (:objects 
      shaker25 - shaker
      left right - hand
      shot24 - shot
      ingredient282 ingredient445 - ingredient
      cocktail273 - cocktail
      dispenser480 dispenser193 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker25)
  (ontable shot24)
  (dispenses dispenser480 ingredient282)
  (dispenses dispenser193 ingredient445)
  (clean shaker25)
  (clean shot24)
  (empty shaker25)
  (empty shot24)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker25 l0)
  (shaker-level shaker25 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail273 ingredient282)
  (cocktail-part2 cocktail273 ingredient445)
)
 (:goal
  (and
      (contains shot24 cocktail273)
)))
