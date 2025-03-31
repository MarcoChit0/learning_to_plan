(define (problem prob)
 (:domain barman)
 (:objects 
      shaker485 - shaker
      left right - hand
      shot460 shot308 - shot
      ingredient237 ingredient101 ingredient338 ingredient151 - ingredient
      cocktail83 - cocktail
      dispenser214 dispenser445 dispenser442 dispenser244 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker485)
  (ontable shot460)
  (ontable shot308)
  (dispenses dispenser214 ingredient237)
  (dispenses dispenser445 ingredient101)
  (dispenses dispenser442 ingredient338)
  (dispenses dispenser244 ingredient151)
  (clean shaker485)
  (clean shot460)
  (clean shot308)
  (empty shaker485)
  (empty shot460)
  (empty shot308)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker485 l0)
  (shaker-level shaker485 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail83 ingredient237)
  (cocktail-part2 cocktail83 ingredient338)
)
 (:goal
  (and
      (contains shot460 cocktail83)
)))
