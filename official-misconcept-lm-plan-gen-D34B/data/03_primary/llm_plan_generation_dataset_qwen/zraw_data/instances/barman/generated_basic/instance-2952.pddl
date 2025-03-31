(define (problem prob)
 (:domain barman)
 (:objects 
      shaker430 - shaker
      left right - hand
      shot150 - shot
      ingredient296 ingredient372 ingredient334 ingredient368 - ingredient
      cocktail371 - cocktail
      dispenser431 dispenser336 dispenser361 dispenser387 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker430)
  (ontable shot150)
  (dispenses dispenser431 ingredient296)
  (dispenses dispenser336 ingredient372)
  (dispenses dispenser361 ingredient334)
  (dispenses dispenser387 ingredient368)
  (clean shaker430)
  (clean shot150)
  (empty shaker430)
  (empty shot150)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker430 l0)
  (shaker-level shaker430 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail371 ingredient372)
  (cocktail-part2 cocktail371 ingredient334)
)
 (:goal
  (and
      (contains shot150 cocktail371)
)))
