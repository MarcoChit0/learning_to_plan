(define (problem prob)
 (:domain barman)
 (:objects 
      shaker384 - shaker
      left right - hand
      shot71 shot417 - shot
      ingredient438 ingredient64 - ingredient
      cocktail1 - cocktail
      dispenser482 dispenser133 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker384)
  (ontable shot71)
  (ontable shot417)
  (dispenses dispenser482 ingredient438)
  (dispenses dispenser133 ingredient64)
  (clean shaker384)
  (clean shot71)
  (clean shot417)
  (empty shaker384)
  (empty shot71)
  (empty shot417)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker384 l0)
  (shaker-level shaker384 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient64)
  (cocktail-part2 cocktail1 ingredient438)
)
 (:goal
  (and
      (contains shot71 cocktail1)
)))
