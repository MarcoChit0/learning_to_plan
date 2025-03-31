(define (problem prob)
 (:domain barman)
 (:objects 
      shaker208 - shaker
      left right - hand
      shot70 shot369 - shot
      ingredient423 ingredient401 ingredient140 - ingredient
      cocktail377 - cocktail
      dispenser435 dispenser48 dispenser42 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker208)
  (ontable shot70)
  (ontable shot369)
  (dispenses dispenser435 ingredient423)
  (dispenses dispenser48 ingredient401)
  (dispenses dispenser42 ingredient140)
  (clean shaker208)
  (clean shot70)
  (clean shot369)
  (empty shaker208)
  (empty shot70)
  (empty shot369)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker208 l0)
  (shaker-level shaker208 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail377 ingredient140)
  (cocktail-part2 cocktail377 ingredient423)
)
 (:goal
  (and
      (contains shot70 cocktail377)
)))
