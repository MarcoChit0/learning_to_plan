(define (problem prob)
 (:domain barman)
 (:objects 
      shaker198 - shaker
      left right - hand
      shot281 - shot
      ingredient113 ingredient420 - ingredient
      cocktail422 - cocktail
      dispenser363 dispenser52 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker198)
  (ontable shot281)
  (dispenses dispenser363 ingredient113)
  (dispenses dispenser52 ingredient420)
  (clean shaker198)
  (clean shot281)
  (empty shaker198)
  (empty shot281)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker198 l0)
  (shaker-level shaker198 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail422 ingredient113)
  (cocktail-part2 cocktail422 ingredient420)
)
 (:goal
  (and
      (contains shot281 cocktail422)
)))
