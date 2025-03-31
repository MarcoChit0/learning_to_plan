(define (problem prob)
 (:domain barman)
 (:objects 
      shaker37 - shaker
      left right - hand
      shot272 - shot
      ingredient447 ingredient494 ingredient12 ingredient301 - ingredient
      cocktail210 - cocktail
      dispenser470 dispenser70 dispenser51 dispenser124 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker37)
  (ontable shot272)
  (dispenses dispenser470 ingredient447)
  (dispenses dispenser70 ingredient494)
  (dispenses dispenser51 ingredient12)
  (dispenses dispenser124 ingredient301)
  (clean shaker37)
  (clean shot272)
  (empty shaker37)
  (empty shot272)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker37 l0)
  (shaker-level shaker37 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail210 ingredient12)
  (cocktail-part2 cocktail210 ingredient301)
)
 (:goal
  (and
      (contains shot272 cocktail210)
)))
