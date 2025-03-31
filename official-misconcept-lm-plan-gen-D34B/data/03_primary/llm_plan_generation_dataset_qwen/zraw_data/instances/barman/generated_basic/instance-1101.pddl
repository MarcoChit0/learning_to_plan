(define (problem prob)
 (:domain barman)
 (:objects 
      shaker431 - shaker
      left right - hand
      shot71 - shot
      ingredient209 ingredient460 ingredient205 - ingredient
      cocktail1 - cocktail
      dispenser251 dispenser368 dispenser438 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker431)
  (ontable shot71)
  (dispenses dispenser251 ingredient209)
  (dispenses dispenser368 ingredient460)
  (dispenses dispenser438 ingredient205)
  (clean shaker431)
  (clean shot71)
  (empty shaker431)
  (empty shot71)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker431 l0)
  (shaker-level shaker431 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient209)
  (cocktail-part2 cocktail1 ingredient205)
)
 (:goal
  (and
      (contains shot71 cocktail1)
)))
