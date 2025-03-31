(define (problem prob)
 (:domain barman)
 (:objects 
      shaker146 - shaker
      left right - hand
      shot461 shot311 - shot
      ingredient155 ingredient272 - ingredient
      cocktail1 - cocktail
      dispenser248 dispenser187 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker146)
  (ontable shot461)
  (ontable shot311)
  (dispenses dispenser248 ingredient155)
  (dispenses dispenser187 ingredient272)
  (clean shaker146)
  (clean shot461)
  (clean shot311)
  (empty shaker146)
  (empty shot461)
  (empty shot311)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker146 l0)
  (shaker-level shaker146 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient272)
  (cocktail-part2 cocktail1 ingredient155)
)
 (:goal
  (and
      (contains shot461 cocktail1)
)))
