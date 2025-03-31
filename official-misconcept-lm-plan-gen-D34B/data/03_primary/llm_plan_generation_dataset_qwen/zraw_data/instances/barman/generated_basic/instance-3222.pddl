(define (problem prob)
 (:domain barman)
 (:objects 
      shaker305 - shaker
      left right - hand
      shot397 - shot
      ingredient291 ingredient499 - ingredient
      cocktail482 - cocktail
      dispenser381 dispenser116 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker305)
  (ontable shot397)
  (dispenses dispenser381 ingredient291)
  (dispenses dispenser116 ingredient499)
  (clean shaker305)
  (clean shot397)
  (empty shaker305)
  (empty shot397)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker305 l0)
  (shaker-level shaker305 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail482 ingredient499)
  (cocktail-part2 cocktail482 ingredient291)
)
 (:goal
  (and
      (contains shot397 cocktail482)
)))
