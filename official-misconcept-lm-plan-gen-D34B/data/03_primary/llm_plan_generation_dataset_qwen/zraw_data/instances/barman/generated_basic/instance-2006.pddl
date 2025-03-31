(define (problem prob)
 (:domain barman)
 (:objects 
      shaker234 - shaker
      left right - hand
      shot446 shot206 shot391 - shot
      ingredient416 ingredient358 ingredient84 - ingredient
      cocktail138 - cocktail
      dispenser419 dispenser476 dispenser87 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker234)
  (ontable shot446)
  (ontable shot206)
  (ontable shot391)
  (dispenses dispenser419 ingredient416)
  (dispenses dispenser476 ingredient358)
  (dispenses dispenser87 ingredient84)
  (clean shaker234)
  (clean shot446)
  (clean shot206)
  (clean shot391)
  (empty shaker234)
  (empty shot446)
  (empty shot206)
  (empty shot391)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker234 l0)
  (shaker-level shaker234 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail138 ingredient416)
  (cocktail-part2 cocktail138 ingredient358)
)
 (:goal
  (and
      (contains shot446 cocktail138)
      (contains shot206 ingredient84)
)))
