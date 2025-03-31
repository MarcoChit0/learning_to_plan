(define (problem prob)
 (:domain barman)
 (:objects 
      shaker363 - shaker
      left right - hand
      shot20 shot259 - shot
      ingredient444 ingredient487 - ingredient
      cocktail78 - cocktail
      dispenser318 dispenser81 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker363)
  (ontable shot20)
  (ontable shot259)
  (dispenses dispenser318 ingredient444)
  (dispenses dispenser81 ingredient487)
  (clean shaker363)
  (clean shot20)
  (clean shot259)
  (empty shaker363)
  (empty shot20)
  (empty shot259)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker363 l0)
  (shaker-level shaker363 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail78 ingredient444)
  (cocktail-part2 cocktail78 ingredient487)
)
 (:goal
  (and
      (contains shot20 cocktail78)
)))
