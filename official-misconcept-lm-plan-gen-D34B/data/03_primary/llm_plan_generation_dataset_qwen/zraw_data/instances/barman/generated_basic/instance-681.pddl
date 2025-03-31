(define (problem prob)
 (:domain barman)
 (:objects 
      shaker433 - shaker
      left right - hand
      shot397 shot227 - shot
      ingredient439 ingredient362 - ingredient
      cocktail1 - cocktail
      dispenser103 dispenser186 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker433)
  (ontable shot397)
  (ontable shot227)
  (dispenses dispenser103 ingredient439)
  (dispenses dispenser186 ingredient362)
  (clean shaker433)
  (clean shot397)
  (clean shot227)
  (empty shaker433)
  (empty shot397)
  (empty shot227)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker433 l0)
  (shaker-level shaker433 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient362)
  (cocktail-part2 cocktail1 ingredient439)
)
 (:goal
  (and
      (contains shot397 cocktail1)
)))
