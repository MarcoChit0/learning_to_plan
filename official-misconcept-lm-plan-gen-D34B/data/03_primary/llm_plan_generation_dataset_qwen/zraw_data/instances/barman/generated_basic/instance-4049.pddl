(define (problem prob)
 (:domain barman)
 (:objects 
      shaker37 - shaker
      left right - hand
      shot8 shot277 - shot
      ingredient173 ingredient395 ingredient53 ingredient310 - ingredient
      cocktail320 - cocktail
      dispenser497 dispenser147 dispenser278 dispenser146 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker37)
  (ontable shot8)
  (ontable shot277)
  (dispenses dispenser497 ingredient173)
  (dispenses dispenser147 ingredient395)
  (dispenses dispenser278 ingredient53)
  (dispenses dispenser146 ingredient310)
  (clean shaker37)
  (clean shot8)
  (clean shot277)
  (empty shaker37)
  (empty shot8)
  (empty shot277)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker37 l0)
  (shaker-level shaker37 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail320 ingredient395)
  (cocktail-part2 cocktail320 ingredient310)
)
 (:goal
  (and
      (contains shot8 cocktail320)
)))
