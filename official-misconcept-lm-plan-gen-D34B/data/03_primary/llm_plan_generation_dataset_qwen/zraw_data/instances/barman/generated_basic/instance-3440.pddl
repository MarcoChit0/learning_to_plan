(define (problem prob)
 (:domain barman)
 (:objects 
      shaker351 - shaker
      left right - hand
      shot213 shot285 - shot
      ingredient467 ingredient150 ingredient303 ingredient426 - ingredient
      cocktail434 - cocktail
      dispenser476 dispenser131 dispenser368 dispenser63 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker351)
  (ontable shot213)
  (ontable shot285)
  (dispenses dispenser476 ingredient467)
  (dispenses dispenser131 ingredient150)
  (dispenses dispenser368 ingredient303)
  (dispenses dispenser63 ingredient426)
  (clean shaker351)
  (clean shot213)
  (clean shot285)
  (empty shaker351)
  (empty shot213)
  (empty shot285)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker351 l0)
  (shaker-level shaker351 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail434 ingredient150)
  (cocktail-part2 cocktail434 ingredient426)
)
 (:goal
  (and
      (contains shot213 cocktail434)
)))
