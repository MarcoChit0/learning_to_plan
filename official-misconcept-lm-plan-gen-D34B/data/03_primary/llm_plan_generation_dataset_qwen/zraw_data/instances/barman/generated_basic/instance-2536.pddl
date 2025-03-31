(define (problem prob)
 (:domain barman)
 (:objects 
      shaker190 - shaker
      left right - hand
      shot494 shot244 - shot
      ingredient63 ingredient159 - ingredient
      cocktail16 - cocktail
      dispenser401 dispenser361 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker190)
  (ontable shot494)
  (ontable shot244)
  (dispenses dispenser401 ingredient63)
  (dispenses dispenser361 ingredient159)
  (clean shaker190)
  (clean shot494)
  (clean shot244)
  (empty shaker190)
  (empty shot494)
  (empty shot244)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker190 l0)
  (shaker-level shaker190 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail16 ingredient63)
  (cocktail-part2 cocktail16 ingredient159)
)
 (:goal
  (and
      (contains shot494 cocktail16)
)))
