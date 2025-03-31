(define (problem prob)
 (:domain barman)
 (:objects 
      shaker94 - shaker
      left right - hand
      shot335 - shot
      ingredient162 ingredient375 ingredient13 ingredient159 - ingredient
      cocktail207 - cocktail
      dispenser102 dispenser250 dispenser323 dispenser126 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker94)
  (ontable shot335)
  (dispenses dispenser102 ingredient162)
  (dispenses dispenser250 ingredient375)
  (dispenses dispenser323 ingredient13)
  (dispenses dispenser126 ingredient159)
  (clean shaker94)
  (clean shot335)
  (empty shaker94)
  (empty shot335)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker94 l0)
  (shaker-level shaker94 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail207 ingredient162)
  (cocktail-part2 cocktail207 ingredient375)
)
 (:goal
  (and
      (contains shot335 cocktail207)
)))
