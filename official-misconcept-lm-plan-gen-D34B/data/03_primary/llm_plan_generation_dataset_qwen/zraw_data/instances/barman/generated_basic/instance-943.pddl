(define (problem prob)
 (:domain barman)
 (:objects 
      shaker131 - shaker
      left right - hand
      shot225 shot156 shot455 - shot
      ingredient355 ingredient404 ingredient236 ingredient259 - ingredient
      cocktail1 - cocktail
      dispenser179 dispenser379 dispenser358 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker131)
  (ontable shot225)
  (ontable shot156)
  (ontable shot455)
  (dispenses dispenser179 ingredient355)
  (dispenses dispenser379 ingredient404)
  (dispenses dispenser358 ingredient236)
  (dispenses dispenser474 ingredient259)
  (clean shaker131)
  (clean shot225)
  (clean shot156)
  (clean shot455)
  (empty shaker131)
  (empty shot225)
  (empty shot156)
  (empty shot455)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker131 l0)
  (shaker-level shaker131 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient404)
  (cocktail-part2 cocktail1 ingredient259)
)
 (:goal
  (and
      (contains shot225 cocktail1)
      (contains shot156 cocktail1)
)))
