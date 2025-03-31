(define (problem prob)
 (:domain barman)
 (:objects 
      shaker195 - shaker
      left right - hand
      shot423 shot432 - shot
      ingredient422 ingredient391 - ingredient
      cocktail416 - cocktail
      dispenser398 dispenser146 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker195)
  (ontable shot423)
  (ontable shot432)
  (dispenses dispenser398 ingredient422)
  (dispenses dispenser146 ingredient391)
  (clean shaker195)
  (clean shot423)
  (clean shot432)
  (empty shaker195)
  (empty shot423)
  (empty shot432)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker195 l0)
  (shaker-level shaker195 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail416 ingredient422)
  (cocktail-part2 cocktail416 ingredient391)
)
 (:goal
  (and
      (contains shot423 cocktail416)
)))
