(define (problem prob)
 (:domain barman)
 (:objects 
      shaker18 - shaker
      left right - hand
      shot111 - shot
      ingredient182 ingredient335 - ingredient
      cocktail164 - cocktail
      dispenser446 dispenser26 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker18)
  (ontable shot111)
  (dispenses dispenser446 ingredient182)
  (dispenses dispenser26 ingredient335)
  (clean shaker18)
  (clean shot111)
  (empty shaker18)
  (empty shot111)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker18 l0)
  (shaker-level shaker18 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail164 ingredient182)
  (cocktail-part2 cocktail164 ingredient335)
)
 (:goal
  (and
      (contains shot111 cocktail164)
)))
