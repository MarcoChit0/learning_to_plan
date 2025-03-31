(define (problem prob)
 (:domain barman)
 (:objects 
      shaker75 - shaker
      left right - hand
      shot463 shot311 - shot
      ingredient317 ingredient365 ingredient426 - ingredient
      cocktail1 - cocktail
      dispenser240 dispenser105 dispenser443 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker75)
  (ontable shot463)
  (ontable shot311)
  (dispenses dispenser240 ingredient317)
  (dispenses dispenser105 ingredient365)
  (dispenses dispenser443 ingredient426)
  (clean shaker75)
  (clean shot463)
  (clean shot311)
  (empty shaker75)
  (empty shot463)
  (empty shot311)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker75 l0)
  (shaker-level shaker75 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient426)
  (cocktail-part2 cocktail1 ingredient365)
)
 (:goal
  (and
      (contains shot463 cocktail1)
)))
