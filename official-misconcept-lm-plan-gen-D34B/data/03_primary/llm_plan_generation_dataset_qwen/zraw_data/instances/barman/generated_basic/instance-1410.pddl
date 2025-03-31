(define (problem prob)
 (:domain barman)
 (:objects 
      shaker360 - shaker
      left right - hand
      shot407 - shot
      ingredient46 ingredient19 ingredient60 ingredient433 - ingredient
      cocktail147 - cocktail
      dispenser285 dispenser322 dispenser299 dispenser282 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker360)
  (ontable shot407)
  (dispenses dispenser285 ingredient46)
  (dispenses dispenser322 ingredient19)
  (dispenses dispenser299 ingredient60)
  (dispenses dispenser282 ingredient433)
  (clean shaker360)
  (clean shot407)
  (empty shaker360)
  (empty shot407)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker360 l0)
  (shaker-level shaker360 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail147 ingredient19)
  (cocktail-part2 cocktail147 ingredient46)
)
 (:goal
  (and
      (contains shot407 cocktail147)
)))
