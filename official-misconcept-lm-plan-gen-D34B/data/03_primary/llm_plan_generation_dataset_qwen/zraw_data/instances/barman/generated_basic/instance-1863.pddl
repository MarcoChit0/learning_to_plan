(define (problem prob)
 (:domain barman)
 (:objects 
      shaker449 - shaker
      left right - hand
      shot212 shot18 - shot
      ingredient54 ingredient222 - ingredient
      cocktail369 - cocktail
      dispenser187 dispenser408 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker449)
  (ontable shot212)
  (ontable shot18)
  (dispenses dispenser187 ingredient54)
  (dispenses dispenser408 ingredient222)
  (clean shaker449)
  (clean shot212)
  (clean shot18)
  (empty shaker449)
  (empty shot212)
  (empty shot18)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker449 l0)
  (shaker-level shaker449 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail369 ingredient222)
  (cocktail-part2 cocktail369 ingredient54)
)
 (:goal
  (and
      (contains shot212 cocktail369)
)))
