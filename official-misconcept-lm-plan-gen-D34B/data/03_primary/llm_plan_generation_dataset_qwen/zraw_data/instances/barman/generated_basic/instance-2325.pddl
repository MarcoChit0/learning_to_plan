(define (problem prob)
 (:domain barman)
 (:objects 
      shaker351 - shaker
      left right - hand
      shot313 - shot
      ingredient68 ingredient402 - ingredient
      cocktail71 - cocktail
      dispenser447 dispenser40 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker351)
  (ontable shot313)
  (dispenses dispenser447 ingredient68)
  (dispenses dispenser40 ingredient402)
  (clean shaker351)
  (clean shot313)
  (empty shaker351)
  (empty shot313)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker351 l0)
  (shaker-level shaker351 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail71 ingredient402)
  (cocktail-part2 cocktail71 ingredient68)
)
 (:goal
  (and
      (contains shot313 cocktail71)
)))
