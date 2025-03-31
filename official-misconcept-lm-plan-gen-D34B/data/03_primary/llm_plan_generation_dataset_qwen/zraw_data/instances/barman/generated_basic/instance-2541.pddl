(define (problem prob)
 (:domain barman)
 (:objects 
      shaker234 - shaker
      left right - hand
      shot349 shot283 - shot
      ingredient108 ingredient325 - ingredient
      cocktail174 - cocktail
      dispenser466 dispenser170 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker234)
  (ontable shot349)
  (ontable shot283)
  (dispenses dispenser466 ingredient108)
  (dispenses dispenser170 ingredient325)
  (clean shaker234)
  (clean shot349)
  (clean shot283)
  (empty shaker234)
  (empty shot349)
  (empty shot283)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker234 l0)
  (shaker-level shaker234 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail174 ingredient108)
  (cocktail-part2 cocktail174 ingredient325)
)
 (:goal
  (and
      (contains shot349 cocktail174)
)))
