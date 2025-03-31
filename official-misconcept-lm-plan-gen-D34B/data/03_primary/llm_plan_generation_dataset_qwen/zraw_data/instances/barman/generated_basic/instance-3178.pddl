(define (problem prob)
 (:domain barman)
 (:objects 
      shaker341 - shaker
      left right - hand
      shot269 - shot
      ingredient219 ingredient379 ingredient410 - ingredient
      cocktail484 - cocktail
      dispenser440 dispenser219 dispenser53 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker341)
  (ontable shot269)
  (dispenses dispenser440 ingredient219)
  (dispenses dispenser219 ingredient379)
  (dispenses dispenser53 ingredient410)
  (clean shaker341)
  (clean shot269)
  (empty shaker341)
  (empty shot269)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker341 l0)
  (shaker-level shaker341 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail484 ingredient410)
  (cocktail-part2 cocktail484 ingredient219)
)
 (:goal
  (and
      (contains shot269 cocktail484)
)))
