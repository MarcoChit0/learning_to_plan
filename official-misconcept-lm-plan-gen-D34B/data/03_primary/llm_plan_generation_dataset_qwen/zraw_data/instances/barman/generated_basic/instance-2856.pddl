(define (problem prob)
 (:domain barman)
 (:objects 
      shaker1 - shaker
      left right - hand
      shot109 shot433 - shot
      ingredient49 ingredient303 ingredient239 ingredient283 - ingredient
      cocktail311 - cocktail
      dispenser367 dispenser492 dispenser273 dispenser332 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker1)
  (ontable shot109)
  (ontable shot433)
  (dispenses dispenser367 ingredient49)
  (dispenses dispenser492 ingredient303)
  (dispenses dispenser273 ingredient239)
  (dispenses dispenser332 ingredient283)
  (clean shaker1)
  (clean shot109)
  (clean shot433)
  (empty shaker1)
  (empty shot109)
  (empty shot433)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker1 l0)
  (shaker-level shaker1 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail311 ingredient303)
  (cocktail-part2 cocktail311 ingredient283)
)
 (:goal
  (and
      (contains shot109 cocktail311)
)))
