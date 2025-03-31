(define (problem prob)
 (:domain barman)
 (:objects 
      shaker128 - shaker
      left right - hand
      shot350 shot478 - shot
      ingredient120 ingredient348 ingredient292 ingredient452 - ingredient
      cocktail361 - cocktail
      dispenser172 dispenser175 dispenser158 dispenser87 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker128)
  (ontable shot350)
  (ontable shot478)
  (dispenses dispenser172 ingredient120)
  (dispenses dispenser175 ingredient348)
  (dispenses dispenser158 ingredient292)
  (dispenses dispenser87 ingredient452)
  (clean shaker128)
  (clean shot350)
  (clean shot478)
  (empty shaker128)
  (empty shot350)
  (empty shot478)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker128 l0)
  (shaker-level shaker128 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail361 ingredient348)
  (cocktail-part2 cocktail361 ingredient292)
)
 (:goal
  (and
      (contains shot350 cocktail361)
)))
