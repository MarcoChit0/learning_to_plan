(define (problem prob)
 (:domain barman)
 (:objects 
      shaker433 - shaker
      left right - hand
      shot86 shot52 - shot
      ingredient414 ingredient467 ingredient245 ingredient430 - ingredient
      cocktail1 - cocktail
      dispenser340 dispenser489 dispenser252 dispenser346 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker433)
  (ontable shot86)
  (ontable shot52)
  (dispenses dispenser340 ingredient414)
  (dispenses dispenser489 ingredient467)
  (dispenses dispenser252 ingredient245)
  (dispenses dispenser346 ingredient430)
  (clean shaker433)
  (clean shot86)
  (clean shot52)
  (empty shaker433)
  (empty shot86)
  (empty shot52)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker433 l0)
  (shaker-level shaker433 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient467)
  (cocktail-part2 cocktail1 ingredient430)
)
 (:goal
  (and
      (contains shot86 cocktail1)
)))
