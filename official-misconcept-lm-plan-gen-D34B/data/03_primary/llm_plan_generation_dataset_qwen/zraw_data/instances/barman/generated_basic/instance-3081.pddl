(define (problem prob)
 (:domain barman)
 (:objects 
      shaker32 - shaker
      left right - hand
      shot108 shot124 - shot
      ingredient38 ingredient451 - ingredient
      cocktail485 - cocktail
      dispenser442 dispenser159 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker32)
  (ontable shot108)
  (ontable shot124)
  (dispenses dispenser442 ingredient38)
  (dispenses dispenser159 ingredient451)
  (clean shaker32)
  (clean shot108)
  (clean shot124)
  (empty shaker32)
  (empty shot108)
  (empty shot124)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker32 l0)
  (shaker-level shaker32 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail485 ingredient38)
  (cocktail-part2 cocktail485 ingredient451)
)
 (:goal
  (and
      (contains shot108 cocktail485)
)))
