(define (problem prob)
 (:domain barman)
 (:objects 
      shaker433 - shaker
      left right - hand
      shot358 - shot
      ingredient69 ingredient30 - ingredient
      cocktail247 - cocktail
      dispenser428 dispenser405 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker433)
  (ontable shot358)
  (dispenses dispenser428 ingredient69)
  (dispenses dispenser405 ingredient30)
  (clean shaker433)
  (clean shot358)
  (empty shaker433)
  (empty shot358)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker433 l0)
  (shaker-level shaker433 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail247 ingredient69)
  (cocktail-part2 cocktail247 ingredient30)
)
 (:goal
  (and
      (contains shot358 cocktail247)
)))
