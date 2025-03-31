(define (problem prob)
 (:domain barman)
 (:objects 
      shaker77 - shaker
      left right - hand
      shot341 - shot
      ingredient410 ingredient354 ingredient39 ingredient159 - ingredient
      cocktail379 - cocktail
      dispenser53 dispenser284 dispenser399 dispenser485 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker77)
  (ontable shot341)
  (dispenses dispenser53 ingredient410)
  (dispenses dispenser284 ingredient354)
  (dispenses dispenser399 ingredient39)
  (dispenses dispenser485 ingredient159)
  (clean shaker77)
  (clean shot341)
  (empty shaker77)
  (empty shot341)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker77 l0)
  (shaker-level shaker77 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail379 ingredient410)
  (cocktail-part2 cocktail379 ingredient354)
)
 (:goal
  (and
      (contains shot341 cocktail379)
)))
