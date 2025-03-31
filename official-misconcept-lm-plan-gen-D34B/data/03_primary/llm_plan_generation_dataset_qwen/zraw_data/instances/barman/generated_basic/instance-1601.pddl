(define (problem prob)
 (:domain barman)
 (:objects 
      shaker150 - shaker
      left right - hand
      shot432 shot165 - shot
      ingredient132 ingredient72 ingredient446 - ingredient
      cocktail486 - cocktail
      dispenser127 dispenser133 dispenser357 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker150)
  (ontable shot432)
  (ontable shot165)
  (dispenses dispenser127 ingredient132)
  (dispenses dispenser133 ingredient72)
  (dispenses dispenser357 ingredient446)
  (clean shaker150)
  (clean shot432)
  (clean shot165)
  (empty shaker150)
  (empty shot432)
  (empty shot165)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker150 l0)
  (shaker-level shaker150 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail486 ingredient446)
  (cocktail-part2 cocktail486 ingredient72)
)
 (:goal
  (and
      (contains shot432 cocktail486)
)))
