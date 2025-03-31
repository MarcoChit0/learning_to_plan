(define (problem prob)
 (:domain barman)
 (:objects 
      shaker41 - shaker
      left right - hand
      shot239 - shot
      ingredient378 ingredient196 - ingredient
      cocktail1 - cocktail
      dispenser374 dispenser343 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker41)
  (ontable shot239)
  (dispenses dispenser374 ingredient378)
  (dispenses dispenser343 ingredient196)
  (clean shaker41)
  (clean shot239)
  (empty shaker41)
  (empty shot239)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker41 l0)
  (shaker-level shaker41 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient378)
  (cocktail-part2 cocktail1 ingredient196)
)
 (:goal
  (and
      (contains shot239 cocktail1)
)))
