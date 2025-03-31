(define (problem prob)
 (:domain barman)
 (:objects 
      shaker257 - shaker
      left right - hand
      shot420 shot306 - shot
      ingredient435 ingredient429 ingredient444 ingredient325 - ingredient
      cocktail83 - cocktail
      dispenser393 dispenser23 dispenser366 dispenser54 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker257)
  (ontable shot420)
  (ontable shot306)
  (dispenses dispenser393 ingredient435)
  (dispenses dispenser23 ingredient429)
  (dispenses dispenser366 ingredient444)
  (dispenses dispenser54 ingredient325)
  (clean shaker257)
  (clean shot420)
  (clean shot306)
  (empty shaker257)
  (empty shot420)
  (empty shot306)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker257 l0)
  (shaker-level shaker257 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail83 ingredient325)
  (cocktail-part2 cocktail83 ingredient444)
)
 (:goal
  (and
      (contains shot420 cocktail83)
)))
