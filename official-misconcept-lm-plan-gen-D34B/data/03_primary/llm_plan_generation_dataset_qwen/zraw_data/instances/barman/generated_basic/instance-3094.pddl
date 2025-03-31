(define (problem prob)
 (:domain barman)
 (:objects 
      shaker341 - shaker
      left right - hand
      shot94 - shot
      ingredient412 ingredient365 ingredient262 ingredient17 - ingredient
      cocktail381 - cocktail
      dispenser35 dispenser135 dispenser308 dispenser210 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker341)
  (ontable shot94)
  (dispenses dispenser35 ingredient412)
  (dispenses dispenser135 ingredient365)
  (dispenses dispenser308 ingredient262)
  (dispenses dispenser210 ingredient17)
  (clean shaker341)
  (clean shot94)
  (empty shaker341)
  (empty shot94)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker341 l0)
  (shaker-level shaker341 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail381 ingredient17)
  (cocktail-part2 cocktail381 ingredient365)
)
 (:goal
  (and
      (contains shot94 cocktail381)
)))
