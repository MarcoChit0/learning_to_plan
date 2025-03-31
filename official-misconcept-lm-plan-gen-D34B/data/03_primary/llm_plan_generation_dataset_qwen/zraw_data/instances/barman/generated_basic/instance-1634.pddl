(define (problem prob)
 (:domain barman)
 (:objects 
      shaker299 - shaker
      left right - hand
      shot458 shot369 shot144 - shot
      ingredient151 ingredient426 - ingredient
      cocktail272 - cocktail
      dispenser103 dispenser320 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker299)
  (ontable shot458)
  (ontable shot369)
  (ontable shot144)
  (dispenses dispenser103 ingredient151)
  (dispenses dispenser320 ingredient426)
  (clean shaker299)
  (clean shot458)
  (clean shot369)
  (clean shot144)
  (empty shaker299)
  (empty shot458)
  (empty shot369)
  (empty shot144)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker299 l0)
  (shaker-level shaker299 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail272 ingredient151)
  (cocktail-part2 cocktail272 ingredient426)
)
 (:goal
  (and
      (contains shot458 cocktail272)
      (contains shot369 cocktail272)
)))
