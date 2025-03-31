(define (problem prob)
 (:domain barman)
 (:objects 
      shaker329 - shaker
      left right - hand
      shot206 shot386 - shot
      ingredient112 ingredient24 - ingredient
      cocktail349 - cocktail
      dispenser64 dispenser243 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker329)
  (ontable shot206)
  (ontable shot386)
  (dispenses dispenser64 ingredient112)
  (dispenses dispenser243 ingredient24)
  (clean shaker329)
  (clean shot206)
  (clean shot386)
  (empty shaker329)
  (empty shot206)
  (empty shot386)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker329 l0)
  (shaker-level shaker329 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail349 ingredient24)
  (cocktail-part2 cocktail349 ingredient112)
)
 (:goal
  (and
      (contains shot206 cocktail349)
)))
