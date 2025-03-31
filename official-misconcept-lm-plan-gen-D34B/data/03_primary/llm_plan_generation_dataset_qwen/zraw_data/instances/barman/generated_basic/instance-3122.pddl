(define (problem prob)
 (:domain barman)
 (:objects 
      shaker229 - shaker
      left right - hand
      shot311 shot131 - shot
      ingredient10 ingredient280 - ingredient
      cocktail297 - cocktail
      dispenser261 dispenser17 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker229)
  (ontable shot311)
  (ontable shot131)
  (dispenses dispenser261 ingredient10)
  (dispenses dispenser17 ingredient280)
  (clean shaker229)
  (clean shot311)
  (clean shot131)
  (empty shaker229)
  (empty shot311)
  (empty shot131)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker229 l0)
  (shaker-level shaker229 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail297 ingredient280)
  (cocktail-part2 cocktail297 ingredient10)
)
 (:goal
  (and
      (contains shot311 cocktail297)
)))
