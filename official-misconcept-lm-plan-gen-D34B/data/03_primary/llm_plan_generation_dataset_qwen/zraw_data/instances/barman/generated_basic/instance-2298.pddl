(define (problem prob)
 (:domain barman)
 (:objects 
      shaker156 - shaker
      left right - hand
      shot389 - shot
      ingredient20 ingredient348 ingredient460 - ingredient
      cocktail352 - cocktail
      dispenser50 dispenser79 dispenser247 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker156)
  (ontable shot389)
  (dispenses dispenser50 ingredient20)
  (dispenses dispenser79 ingredient348)
  (dispenses dispenser247 ingredient460)
  (clean shaker156)
  (clean shot389)
  (empty shaker156)
  (empty shot389)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker156 l0)
  (shaker-level shaker156 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail352 ingredient348)
  (cocktail-part2 cocktail352 ingredient460)
)
 (:goal
  (and
      (contains shot389 cocktail352)
)))
