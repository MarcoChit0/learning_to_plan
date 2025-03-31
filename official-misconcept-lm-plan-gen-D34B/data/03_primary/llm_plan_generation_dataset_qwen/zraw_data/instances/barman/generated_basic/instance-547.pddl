(define (problem prob)
 (:domain barman)
 (:objects 
      shaker332 - shaker
      left right - hand
      shot378 - shot
      ingredient436 ingredient288 ingredient349 - ingredient
      cocktail1 - cocktail
      dispenser430 dispenser164 dispenser131 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker332)
  (ontable shot378)
  (dispenses dispenser430 ingredient436)
  (dispenses dispenser164 ingredient288)
  (dispenses dispenser131 ingredient349)
  (clean shaker332)
  (clean shot378)
  (empty shaker332)
  (empty shot378)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker332 l0)
  (shaker-level shaker332 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient436)
  (cocktail-part2 cocktail1 ingredient349)
)
 (:goal
  (and
      (contains shot378 cocktail1)
)))
