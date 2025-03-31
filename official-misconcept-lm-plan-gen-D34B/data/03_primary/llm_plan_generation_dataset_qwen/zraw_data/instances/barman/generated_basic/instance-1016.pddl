(define (problem prob)
 (:domain barman)
 (:objects 
      shaker441 - shaker
      left right - hand
      shot214 - shot
      ingredient402 ingredient138 ingredient438 - ingredient
      cocktail1 - cocktail
      dispenser339 dispenser181 dispenser87 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker441)
  (ontable shot214)
  (dispenses dispenser339 ingredient402)
  (dispenses dispenser181 ingredient138)
  (dispenses dispenser87 ingredient438)
  (clean shaker441)
  (clean shot214)
  (empty shaker441)
  (empty shot214)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker441 l0)
  (shaker-level shaker441 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient402)
  (cocktail-part2 cocktail1 ingredient138)
)
 (:goal
  (and
      (contains shot214 cocktail1)
)))
