(define (problem prob)
 (:domain barman)
 (:objects 
      shaker499 - shaker
      left right - hand
      shot58 - shot
      ingredient42 ingredient268 ingredient196 - ingredient
      cocktail221 - cocktail
      dispenser442 dispenser342 dispenser274 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker499)
  (ontable shot58)
  (dispenses dispenser442 ingredient42)
  (dispenses dispenser342 ingredient268)
  (dispenses dispenser274 ingredient196)
  (clean shaker499)
  (clean shot58)
  (empty shaker499)
  (empty shot58)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker499 l0)
  (shaker-level shaker499 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail221 ingredient268)
  (cocktail-part2 cocktail221 ingredient42)
)
 (:goal
  (and
      (contains shot58 cocktail221)
)))
