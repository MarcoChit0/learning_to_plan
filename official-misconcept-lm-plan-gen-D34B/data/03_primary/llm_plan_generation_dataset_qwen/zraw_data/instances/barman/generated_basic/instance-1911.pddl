(define (problem prob)
 (:domain barman)
 (:objects 
      shaker437 - shaker
      left right - hand
      shot320 - shot
      ingredient284 ingredient73 ingredient312 ingredient366 - ingredient
      cocktail185 - cocktail
      dispenser123 dispenser20 dispenser86 dispenser410 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker437)
  (ontable shot320)
  (dispenses dispenser123 ingredient284)
  (dispenses dispenser20 ingredient73)
  (dispenses dispenser86 ingredient312)
  (dispenses dispenser410 ingredient366)
  (clean shaker437)
  (clean shot320)
  (empty shaker437)
  (empty shot320)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker437 l0)
  (shaker-level shaker437 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail185 ingredient73)
  (cocktail-part2 cocktail185 ingredient284)
)
 (:goal
  (and
      (contains shot320 cocktail185)
)))
