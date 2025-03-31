(define (problem prob)
 (:domain barman)
 (:objects 
      shaker209 - shaker
      left right - hand
      shot91 shot61 - shot
      ingredient66 ingredient117 ingredient173 ingredient115 - ingredient
      cocktail1 - cocktail
      dispenser345 dispenser56 dispenser238 dispenser464 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker209)
  (ontable shot91)
  (ontable shot61)
  (dispenses dispenser345 ingredient66)
  (dispenses dispenser56 ingredient117)
  (dispenses dispenser238 ingredient173)
  (dispenses dispenser464 ingredient115)
  (clean shaker209)
  (clean shot91)
  (clean shot61)
  (empty shaker209)
  (empty shot91)
  (empty shot61)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker209 l0)
  (shaker-level shaker209 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient173)
  (cocktail-part2 cocktail1 ingredient117)
)
 (:goal
  (and
      (contains shot91 cocktail1)
)))
