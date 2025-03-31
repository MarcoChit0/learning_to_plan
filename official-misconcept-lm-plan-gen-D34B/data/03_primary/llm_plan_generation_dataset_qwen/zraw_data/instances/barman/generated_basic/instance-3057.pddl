(define (problem prob)
 (:domain barman)
 (:objects 
      shaker111 - shaker
      left right - hand
      shot385 shot66 - shot
      ingredient41 ingredient230 ingredient3 - ingredient
      cocktail110 - cocktail
      dispenser69 dispenser286 dispenser30 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker111)
  (ontable shot385)
  (ontable shot66)
  (dispenses dispenser69 ingredient41)
  (dispenses dispenser286 ingredient230)
  (dispenses dispenser30 ingredient3)
  (clean shaker111)
  (clean shot385)
  (clean shot66)
  (empty shaker111)
  (empty shot385)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker111 l0)
  (shaker-level shaker111 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail110 ingredient3)
  (cocktail-part2 cocktail110 ingredient230)
)
 (:goal
  (and
      (contains shot385 cocktail110)
)))
