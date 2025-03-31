(define (problem prob)
 (:domain barman)
 (:objects 
      shaker180 - shaker
      left right - hand
      shot422 - shot
      ingredient8 ingredient447 ingredient64 - ingredient
      cocktail1 - cocktail
      dispenser369 dispenser382 dispenser309 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker180)
  (ontable shot422)
  (dispenses dispenser369 ingredient8)
  (dispenses dispenser382 ingredient447)
  (dispenses dispenser309 ingredient64)
  (clean shaker180)
  (clean shot422)
  (empty shaker180)
  (empty shot422)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker180 l0)
  (shaker-level shaker180 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient8)
  (cocktail-part2 cocktail1 ingredient64)
)
 (:goal
  (and
      (contains shot422 cocktail1)
)))
