(define (problem prob)
 (:domain barman)
 (:objects 
      shaker294 - shaker
      left right - hand
      shot380 - shot
      ingredient105 ingredient142 - ingredient
      cocktail371 - cocktail
      dispenser392 dispenser93 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker294)
  (ontable shot380)
  (dispenses dispenser392 ingredient105)
  (dispenses dispenser93 ingredient142)
  (clean shaker294)
  (clean shot380)
  (empty shaker294)
  (empty shot380)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker294 l0)
  (shaker-level shaker294 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail371 ingredient142)
  (cocktail-part2 cocktail371 ingredient105)
)
 (:goal
  (and
      (contains shot380 cocktail371)
)))
