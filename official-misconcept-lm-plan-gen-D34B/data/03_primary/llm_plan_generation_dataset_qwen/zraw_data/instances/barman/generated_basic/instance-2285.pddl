(define (problem prob)
 (:domain barman)
 (:objects 
      shaker400 - shaker
      left right - hand
      shot66 - shot
      ingredient403 ingredient477 ingredient471 - ingredient
      cocktail414 - cocktail
      dispenser403 dispenser156 dispenser141 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker400)
  (ontable shot66)
  (dispenses dispenser403 ingredient403)
  (dispenses dispenser156 ingredient477)
  (dispenses dispenser141 ingredient471)
  (clean shaker400)
  (clean shot66)
  (empty shaker400)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker400 l0)
  (shaker-level shaker400 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail414 ingredient477)
  (cocktail-part2 cocktail414 ingredient403)
)
 (:goal
  (and
      (contains shot66 cocktail414)
)))
