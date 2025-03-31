(define (problem prob)
 (:domain barman)
 (:objects 
      shaker149 - shaker
      left right - hand
      shot222 shot107 - shot
      ingredient268 ingredient200 ingredient412 ingredient120 - ingredient
      cocktail1 - cocktail
      dispenser146 dispenser372 dispenser232 dispenser330 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker149)
  (ontable shot222)
  (ontable shot107)
  (dispenses dispenser146 ingredient268)
  (dispenses dispenser372 ingredient200)
  (dispenses dispenser232 ingredient412)
  (dispenses dispenser330 ingredient120)
  (clean shaker149)
  (clean shot222)
  (clean shot107)
  (empty shaker149)
  (empty shot222)
  (empty shot107)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker149 l0)
  (shaker-level shaker149 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient200)
  (cocktail-part2 cocktail1 ingredient412)
)
 (:goal
  (and
      (contains shot222 cocktail1)
)))
