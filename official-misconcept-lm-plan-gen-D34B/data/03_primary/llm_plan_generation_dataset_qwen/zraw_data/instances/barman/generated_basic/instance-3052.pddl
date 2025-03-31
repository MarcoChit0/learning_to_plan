(define (problem prob)
 (:domain barman)
 (:objects 
      shaker439 - shaker
      left right - hand
      shot369 shot73 - shot
      ingredient266 ingredient235 ingredient78 - ingredient
      cocktail179 - cocktail
      dispenser64 dispenser126 dispenser328 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker439)
  (ontable shot369)
  (ontable shot73)
  (dispenses dispenser64 ingredient266)
  (dispenses dispenser126 ingredient235)
  (dispenses dispenser328 ingredient78)
  (clean shaker439)
  (clean shot369)
  (clean shot73)
  (empty shaker439)
  (empty shot369)
  (empty shot73)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker439 l0)
  (shaker-level shaker439 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail179 ingredient266)
  (cocktail-part2 cocktail179 ingredient235)
)
 (:goal
  (and
      (contains shot369 cocktail179)
)))
