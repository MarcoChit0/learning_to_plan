(define (problem prob)
 (:domain barman)
 (:objects 
      shaker38 - shaker
      left right - hand
      shot14 shot492 - shot
      ingredient150 ingredient80 ingredient234 ingredient117 - ingredient
      cocktail279 - cocktail
      dispenser80 dispenser95 dispenser387 dispenser192 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker38)
  (ontable shot14)
  (ontable shot492)
  (dispenses dispenser80 ingredient150)
  (dispenses dispenser95 ingredient80)
  (dispenses dispenser387 ingredient234)
  (dispenses dispenser192 ingredient117)
  (clean shaker38)
  (clean shot14)
  (clean shot492)
  (empty shaker38)
  (empty shot14)
  (empty shot492)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker38 l0)
  (shaker-level shaker38 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail279 ingredient117)
  (cocktail-part2 cocktail279 ingredient150)
)
 (:goal
  (and
      (contains shot14 cocktail279)
)))
