(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot492 - shot
      ingredient311 ingredient113 - ingredient
      cocktail390 - cocktail
      dispenser266 dispenser318 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot492)
  (dispenses dispenser266 ingredient311)
  (dispenses dispenser318 ingredient113)
  (clean shaker221)
  (clean shot492)
  (empty shaker221)
  (empty shot492)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail390 ingredient113)
  (cocktail-part2 cocktail390 ingredient311)
)
 (:goal
  (and
      (contains shot492 cocktail390)
)))
