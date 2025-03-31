(define (problem prob)
 (:domain barman)
 (:objects 
      shaker496 - shaker
      left right - hand
      shot209 - shot
      ingredient390 ingredient90 - ingredient
      cocktail27 - cocktail
      dispenser139 dispenser22 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker496)
  (ontable shot209)
  (dispenses dispenser139 ingredient390)
  (dispenses dispenser22 ingredient90)
  (clean shaker496)
  (clean shot209)
  (empty shaker496)
  (empty shot209)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker496 l0)
  (shaker-level shaker496 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail27 ingredient390)
  (cocktail-part2 cocktail27 ingredient90)
)
 (:goal
  (and
      (contains shot209 cocktail27)
)))
