(define (problem prob)
 (:domain barman)
 (:objects 
      shaker323 - shaker
      left right - hand
      shot484 shot57 - shot
      ingredient352 ingredient188 ingredient155 - ingredient
      cocktail346 - cocktail
      dispenser286 dispenser183 dispenser466 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker323)
  (ontable shot484)
  (ontable shot57)
  (dispenses dispenser286 ingredient352)
  (dispenses dispenser183 ingredient188)
  (dispenses dispenser466 ingredient155)
  (clean shaker323)
  (clean shot484)
  (clean shot57)
  (empty shaker323)
  (empty shot484)
  (empty shot57)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker323 l0)
  (shaker-level shaker323 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail346 ingredient188)
  (cocktail-part2 cocktail346 ingredient352)
)
 (:goal
  (and
      (contains shot484 cocktail346)
)))
