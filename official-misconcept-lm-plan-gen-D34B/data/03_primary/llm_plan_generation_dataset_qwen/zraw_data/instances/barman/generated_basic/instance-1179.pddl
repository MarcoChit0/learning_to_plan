(define (problem prob)
 (:domain barman)
 (:objects 
      shaker227 - shaker
      left right - hand
      shot330 shot245 - shot
      ingredient438 ingredient148 ingredient437 - ingredient
      cocktail1 - cocktail
      dispenser329 dispenser309 dispenser162 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker227)
  (ontable shot330)
  (ontable shot245)
  (dispenses dispenser329 ingredient438)
  (dispenses dispenser309 ingredient148)
  (dispenses dispenser162 ingredient437)
  (clean shaker227)
  (clean shot330)
  (clean shot245)
  (empty shaker227)
  (empty shot330)
  (empty shot245)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker227 l0)
  (shaker-level shaker227 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient438)
  (cocktail-part2 cocktail1 ingredient437)
)
 (:goal
  (and
      (contains shot330 cocktail1)
)))
