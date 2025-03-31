(define (problem prob)
 (:domain barman)
 (:objects 
      shaker31 - shaker
      left right - hand
      shot275 - shot
      ingredient167 ingredient358 ingredient18 ingredient123 - ingredient
      cocktail4 - cocktail
      dispenser483 dispenser343 dispenser171 dispenser226 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker31)
  (ontable shot275)
  (dispenses dispenser483 ingredient167)
  (dispenses dispenser343 ingredient358)
  (dispenses dispenser171 ingredient18)
  (dispenses dispenser226 ingredient123)
  (clean shaker31)
  (clean shot275)
  (empty shaker31)
  (empty shot275)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker31 l0)
  (shaker-level shaker31 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail4 ingredient123)
  (cocktail-part2 cocktail4 ingredient358)
)
 (:goal
  (and
      (contains shot275 cocktail4)
)))
