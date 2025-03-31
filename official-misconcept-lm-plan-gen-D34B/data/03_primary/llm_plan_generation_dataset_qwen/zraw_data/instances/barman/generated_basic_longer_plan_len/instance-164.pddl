(define (problem prob)
 (:domain barman)
 (:objects 
      shaker298 - shaker
      left right - hand
      shot405 shot384 shot70 shot334 - shot
      ingredient251 ingredient0 ingredient25 ingredient44 - ingredient
      cocktail1 - cocktail
      dispenser406 dispenser139 dispenser244 dispenser248 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker298)
  (ontable shot405)
  (ontable shot384)
  (ontable shot70)
  (ontable shot334)
  (dispenses dispenser406 ingredient251)
  (dispenses dispenser139 ingredient0)
  (dispenses dispenser244 ingredient25)
  (dispenses dispenser248 ingredient44)
  (clean shaker298)
  (clean shot405)
  (clean shot384)
  (clean shot70)
  (clean shot334)
  (empty shaker298)
  (empty shot405)
  (empty shot384)
  (empty shot70)
  (empty shot334)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker298 l0)
  (shaker-level shaker298 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient0)
  (cocktail-part2 cocktail1 ingredient44)
)
 (:goal
  (and
      (contains shot405 cocktail1)
      (contains shot384 ingredient251)
      (contains shot70 ingredient0)
)))
