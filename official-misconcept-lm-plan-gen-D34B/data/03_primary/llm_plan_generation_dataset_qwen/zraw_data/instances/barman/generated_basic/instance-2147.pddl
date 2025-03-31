(define (problem prob)
 (:domain barman)
 (:objects 
      shaker2 - shaker
      left right - hand
      shot155 - shot
      ingredient226 ingredient136 - ingredient
      cocktail483 - cocktail
      dispenser290 dispenser288 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker2)
  (ontable shot155)
  (dispenses dispenser290 ingredient226)
  (dispenses dispenser288 ingredient136)
  (clean shaker2)
  (clean shot155)
  (empty shaker2)
  (empty shot155)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker2 l0)
  (shaker-level shaker2 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail483 ingredient226)
  (cocktail-part2 cocktail483 ingredient136)
)
 (:goal
  (and
      (contains shot155 cocktail483)
)))
