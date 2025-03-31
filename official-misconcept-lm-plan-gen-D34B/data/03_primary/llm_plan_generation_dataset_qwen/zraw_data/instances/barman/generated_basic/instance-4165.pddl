(define (problem prob)
 (:domain barman)
 (:objects 
      shaker211 - shaker
      left right - hand
      shot6 shot37 - shot
      ingredient329 ingredient216 - ingredient
      cocktail232 - cocktail
      dispenser390 dispenser384 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker211)
  (ontable shot6)
  (ontable shot37)
  (dispenses dispenser390 ingredient329)
  (dispenses dispenser384 ingredient216)
  (clean shaker211)
  (clean shot6)
  (clean shot37)
  (empty shaker211)
  (empty shot6)
  (empty shot37)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker211 l0)
  (shaker-level shaker211 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail232 ingredient216)
  (cocktail-part2 cocktail232 ingredient329)
)
 (:goal
  (and
      (contains shot6 cocktail232)
)))
