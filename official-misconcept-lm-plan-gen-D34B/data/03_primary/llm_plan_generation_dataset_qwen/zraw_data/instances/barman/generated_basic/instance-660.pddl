(define (problem prob)
 (:domain barman)
 (:objects 
      shaker295 - shaker
      left right - hand
      shot164 shot178 - shot
      ingredient376 ingredient49 ingredient477 - ingredient
      cocktail1 - cocktail
      dispenser359 dispenser402 dispenser187 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker295)
  (ontable shot164)
  (ontable shot178)
  (dispenses dispenser359 ingredient376)
  (dispenses dispenser402 ingredient49)
  (dispenses dispenser187 ingredient477)
  (clean shaker295)
  (clean shot164)
  (clean shot178)
  (empty shaker295)
  (empty shot164)
  (empty shot178)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker295 l0)
  (shaker-level shaker295 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient49)
  (cocktail-part2 cocktail1 ingredient376)
)
 (:goal
  (and
      (contains shot164 cocktail1)
)))
