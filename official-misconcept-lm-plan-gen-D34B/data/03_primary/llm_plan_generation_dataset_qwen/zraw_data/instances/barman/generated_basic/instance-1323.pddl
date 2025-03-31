(define (problem prob)
 (:domain barman)
 (:objects 
      shaker256 - shaker
      left right - hand
      shot339 - shot
      ingredient294 ingredient9 ingredient344 ingredient409 - ingredient
      cocktail290 - cocktail
      dispenser371 dispenser16 dispenser407 dispenser310 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker256)
  (ontable shot339)
  (dispenses dispenser371 ingredient294)
  (dispenses dispenser16 ingredient9)
  (dispenses dispenser407 ingredient344)
  (dispenses dispenser310 ingredient409)
  (clean shaker256)
  (clean shot339)
  (empty shaker256)
  (empty shot339)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker256 l0)
  (shaker-level shaker256 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail290 ingredient344)
  (cocktail-part2 cocktail290 ingredient294)
)
 (:goal
  (and
      (contains shot339 cocktail290)
)))
