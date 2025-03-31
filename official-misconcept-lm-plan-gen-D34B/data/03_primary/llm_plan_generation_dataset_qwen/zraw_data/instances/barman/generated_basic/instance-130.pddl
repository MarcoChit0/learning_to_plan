(define (problem prob)
 (:domain barman)
 (:objects 
      shaker296 - shaker
      left right - hand
      shot307 shot279 - shot
      ingredient363 ingredient2 - ingredient
      cocktail1 - cocktail
      dispenser194 dispenser52 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker296)
  (ontable shot307)
  (ontable shot279)
  (dispenses dispenser194 ingredient363)
  (dispenses dispenser52 ingredient2)
  (clean shaker296)
  (clean shot307)
  (clean shot279)
  (empty shaker296)
  (empty shot307)
  (empty shot279)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker296 l0)
  (shaker-level shaker296 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient2)
  (cocktail-part2 cocktail1 ingredient363)
)
 (:goal
  (and
      (contains shot307 cocktail1)
)))
