(define (problem prob)
 (:domain barman)
 (:objects 
      shaker344 - shaker
      left right - hand
      shot37 shot79 - shot
      ingredient411 ingredient216 - ingredient
      cocktail1 - cocktail
      dispenser159 dispenser166 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker344)
  (ontable shot37)
  (ontable shot79)
  (dispenses dispenser159 ingredient411)
  (dispenses dispenser166 ingredient216)
  (clean shaker344)
  (clean shot37)
  (clean shot79)
  (empty shaker344)
  (empty shot37)
  (empty shot79)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker344 l0)
  (shaker-level shaker344 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient216)
  (cocktail-part2 cocktail1 ingredient411)
)
 (:goal
  (and
      (contains shot37 cocktail1)
)))
