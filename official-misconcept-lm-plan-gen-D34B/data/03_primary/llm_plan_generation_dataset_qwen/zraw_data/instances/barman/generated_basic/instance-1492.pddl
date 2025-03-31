(define (problem prob)
 (:domain barman)
 (:objects 
      shaker2 - shaker
      left right - hand
      shot329 shot70 - shot
      ingredient282 ingredient218 ingredient312 ingredient189 - ingredient
      cocktail77 - cocktail
      dispenser414 dispenser237 dispenser269 dispenser306 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker2)
  (ontable shot329)
  (ontable shot70)
  (dispenses dispenser414 ingredient282)
  (dispenses dispenser237 ingredient218)
  (dispenses dispenser269 ingredient312)
  (dispenses dispenser306 ingredient189)
  (clean shaker2)
  (clean shot329)
  (clean shot70)
  (empty shaker2)
  (empty shot329)
  (empty shot70)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker2 l0)
  (shaker-level shaker2 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail77 ingredient218)
  (cocktail-part2 cocktail77 ingredient189)
)
 (:goal
  (and
      (contains shot329 cocktail77)
)))
