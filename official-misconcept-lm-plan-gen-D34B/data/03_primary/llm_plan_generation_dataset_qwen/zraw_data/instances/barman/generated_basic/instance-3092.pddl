(define (problem prob)
 (:domain barman)
 (:objects 
      shaker114 - shaker
      left right - hand
      shot140 - shot
      ingredient142 ingredient108 ingredient368 ingredient16 - ingredient
      cocktail194 - cocktail
      dispenser53 dispenser107 dispenser329 dispenser320 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker114)
  (ontable shot140)
  (dispenses dispenser53 ingredient142)
  (dispenses dispenser107 ingredient108)
  (dispenses dispenser329 ingredient368)
  (dispenses dispenser320 ingredient16)
  (clean shaker114)
  (clean shot140)
  (empty shaker114)
  (empty shot140)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker114 l0)
  (shaker-level shaker114 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail194 ingredient142)
  (cocktail-part2 cocktail194 ingredient108)
)
 (:goal
  (and
      (contains shot140 cocktail194)
)))
