(define (problem prob)
 (:domain barman)
 (:objects 
      shaker451 - shaker
      left right - hand
      shot441 - shot
      ingredient442 ingredient439 ingredient327 - ingredient
      cocktail260 - cocktail
      dispenser318 dispenser312 dispenser487 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker451)
  (ontable shot441)
  (dispenses dispenser318 ingredient442)
  (dispenses dispenser312 ingredient439)
  (dispenses dispenser487 ingredient327)
  (clean shaker451)
  (clean shot441)
  (empty shaker451)
  (empty shot441)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker451 l0)
  (shaker-level shaker451 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail260 ingredient439)
  (cocktail-part2 cocktail260 ingredient327)
)
 (:goal
  (and
      (contains shot441 cocktail260)
)))
