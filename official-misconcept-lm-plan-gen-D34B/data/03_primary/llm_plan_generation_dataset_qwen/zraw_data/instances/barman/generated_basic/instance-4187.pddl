(define (problem prob)
 (:domain barman)
 (:objects 
      shaker427 - shaker
      left right - hand
      shot424 - shot
      ingredient473 ingredient423 ingredient469 ingredient484 - ingredient
      cocktail0 - cocktail
      dispenser136 dispenser18 dispenser402 dispenser30 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker427)
  (ontable shot424)
  (dispenses dispenser136 ingredient473)
  (dispenses dispenser18 ingredient423)
  (dispenses dispenser402 ingredient469)
  (dispenses dispenser30 ingredient484)
  (clean shaker427)
  (clean shot424)
  (empty shaker427)
  (empty shot424)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker427 l0)
  (shaker-level shaker427 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail0 ingredient423)
  (cocktail-part2 cocktail0 ingredient469)
)
 (:goal
  (and
      (contains shot424 cocktail0)
)))
