(define (problem prob)
 (:domain barman)
 (:objects 
      shaker212 - shaker
      left right - hand
      shot267 shot382 shot474 - shot
      ingredient159 ingredient439 ingredient47 - ingredient
      cocktail354 - cocktail
      dispenser200 dispenser240 dispenser109 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker212)
  (ontable shot267)
  (ontable shot382)
  (ontable shot474)
  (dispenses dispenser200 ingredient159)
  (dispenses dispenser240 ingredient439)
  (dispenses dispenser109 ingredient47)
  (clean shaker212)
  (clean shot267)
  (clean shot382)
  (clean shot474)
  (empty shaker212)
  (empty shot267)
  (empty shot382)
  (empty shot474)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker212 l0)
  (shaker-level shaker212 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail354 ingredient439)
  (cocktail-part2 cocktail354 ingredient159)
)
 (:goal
  (and
      (contains shot267 cocktail354)
      (contains shot382 ingredient439)
)))
