(define (problem prob)
 (:domain barman)
 (:objects 
      shaker9 - shaker
      left right - hand
      shot334 shot316 - shot
      ingredient43 ingredient406 - ingredient
      cocktail300 - cocktail
      dispenser10 dispenser302 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker9)
  (ontable shot334)
  (ontable shot316)
  (dispenses dispenser10 ingredient43)
  (dispenses dispenser302 ingredient406)
  (clean shaker9)
  (clean shot334)
  (clean shot316)
  (empty shaker9)
  (empty shot334)
  (empty shot316)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker9 l0)
  (shaker-level shaker9 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail300 ingredient406)
  (cocktail-part2 cocktail300 ingredient43)
)
 (:goal
  (and
      (contains shot334 cocktail300)
)))
