(define (problem prob)
 (:domain barman)
 (:objects 
      shaker290 - shaker
      left right - hand
      shot156 - shot
      ingredient356 ingredient128 - ingredient
      cocktail274 - cocktail
      dispenser468 dispenser408 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker290)
  (ontable shot156)
  (dispenses dispenser468 ingredient356)
  (dispenses dispenser408 ingredient128)
  (clean shaker290)
  (clean shot156)
  (empty shaker290)
  (empty shot156)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker290 l0)
  (shaker-level shaker290 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail274 ingredient356)
  (cocktail-part2 cocktail274 ingredient128)
)
 (:goal
  (and
      (contains shot156 cocktail274)
)))
