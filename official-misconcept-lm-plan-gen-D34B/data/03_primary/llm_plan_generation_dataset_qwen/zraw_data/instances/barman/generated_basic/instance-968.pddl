(define (problem prob)
 (:domain barman)
 (:objects 
      shaker382 - shaker
      left right - hand
      shot91 - shot
      ingredient446 ingredient57 ingredient423 ingredient272 - ingredient
      cocktail1 - cocktail
      dispenser185 dispenser55 dispenser100 dispenser127 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker382)
  (ontable shot91)
  (dispenses dispenser185 ingredient446)
  (dispenses dispenser55 ingredient57)
  (dispenses dispenser100 ingredient423)
  (dispenses dispenser127 ingredient272)
  (clean shaker382)
  (clean shot91)
  (empty shaker382)
  (empty shot91)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker382 l0)
  (shaker-level shaker382 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient446)
  (cocktail-part2 cocktail1 ingredient57)
)
 (:goal
  (and
      (contains shot91 cocktail1)
)))
