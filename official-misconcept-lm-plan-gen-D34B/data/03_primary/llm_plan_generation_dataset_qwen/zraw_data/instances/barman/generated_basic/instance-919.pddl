(define (problem prob)
 (:domain barman)
 (:objects 
      shaker70 - shaker
      left right - hand
      shot193 shot472 - shot
      ingredient189 ingredient445 ingredient85 - ingredient
      cocktail1 - cocktail
      dispenser313 dispenser139 dispenser343 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker70)
  (ontable shot193)
  (ontable shot472)
  (dispenses dispenser313 ingredient189)
  (dispenses dispenser139 ingredient445)
  (dispenses dispenser343 ingredient85)
  (clean shaker70)
  (clean shot193)
  (clean shot472)
  (empty shaker70)
  (empty shot193)
  (empty shot472)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker70 l0)
  (shaker-level shaker70 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient189)
  (cocktail-part2 cocktail1 ingredient445)
)
 (:goal
  (and
      (contains shot193 cocktail1)
)))
