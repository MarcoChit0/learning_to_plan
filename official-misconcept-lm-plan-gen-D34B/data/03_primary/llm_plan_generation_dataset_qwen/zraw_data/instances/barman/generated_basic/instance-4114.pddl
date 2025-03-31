(define (problem prob)
 (:domain barman)
 (:objects 
      shaker1 - shaker
      left right - hand
      shot342 shot431 - shot
      ingredient411 ingredient79 - ingredient
      cocktail434 - cocktail
      dispenser497 dispenser402 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker1)
  (ontable shot342)
  (ontable shot431)
  (dispenses dispenser497 ingredient411)
  (dispenses dispenser402 ingredient79)
  (clean shaker1)
  (clean shot342)
  (clean shot431)
  (empty shaker1)
  (empty shot342)
  (empty shot431)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker1 l0)
  (shaker-level shaker1 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail434 ingredient79)
  (cocktail-part2 cocktail434 ingredient411)
)
 (:goal
  (and
      (contains shot342 cocktail434)
)))
