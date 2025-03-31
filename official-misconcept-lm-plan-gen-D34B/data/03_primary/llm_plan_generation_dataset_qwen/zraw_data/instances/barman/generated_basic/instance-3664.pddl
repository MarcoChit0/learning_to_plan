(define (problem prob)
 (:domain barman)
 (:objects 
      shaker162 - shaker
      left right - hand
      shot205 shot337 - shot
      ingredient311 ingredient470 ingredient3 ingredient129 - ingredient
      cocktail396 - cocktail
      dispenser259 dispenser476 dispenser377 dispenser40 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker162)
  (ontable shot205)
  (ontable shot337)
  (dispenses dispenser259 ingredient311)
  (dispenses dispenser476 ingredient470)
  (dispenses dispenser377 ingredient3)
  (dispenses dispenser40 ingredient129)
  (clean shaker162)
  (clean shot205)
  (clean shot337)
  (empty shaker162)
  (empty shot205)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker162 l0)
  (shaker-level shaker162 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail396 ingredient311)
  (cocktail-part2 cocktail396 ingredient129)
)
 (:goal
  (and
      (contains shot205 cocktail396)
)))
