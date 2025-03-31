(define (problem prob)
 (:domain barman)
 (:objects 
      shaker410 - shaker
      left right - hand
      shot330 shot411 - shot
      ingredient228 ingredient66 - ingredient
      cocktail1 - cocktail
      dispenser78 dispenser489 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker410)
  (ontable shot330)
  (ontable shot411)
  (dispenses dispenser78 ingredient228)
  (dispenses dispenser489 ingredient66)
  (clean shaker410)
  (clean shot330)
  (clean shot411)
  (empty shaker410)
  (empty shot330)
  (empty shot411)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker410 l0)
  (shaker-level shaker410 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient228)
  (cocktail-part2 cocktail1 ingredient66)
)
 (:goal
  (and
      (contains shot330 cocktail1)
)))
