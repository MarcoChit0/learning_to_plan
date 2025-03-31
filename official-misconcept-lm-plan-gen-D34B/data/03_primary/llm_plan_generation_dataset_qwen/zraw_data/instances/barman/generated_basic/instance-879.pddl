(define (problem prob)
 (:domain barman)
 (:objects 
      shaker51 - shaker
      left right - hand
      shot431 - shot
      ingredient229 ingredient94 - ingredient
      cocktail1 - cocktail
      dispenser444 dispenser220 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker51)
  (ontable shot431)
  (dispenses dispenser444 ingredient229)
  (dispenses dispenser220 ingredient94)
  (clean shaker51)
  (clean shot431)
  (empty shaker51)
  (empty shot431)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker51 l0)
  (shaker-level shaker51 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient229)
  (cocktail-part2 cocktail1 ingredient94)
)
 (:goal
  (and
      (contains shot431 cocktail1)
)))
