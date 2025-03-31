(define (problem prob)
 (:domain barman)
 (:objects 
      shaker326 - shaker
      left right - hand
      shot2 shot217 shot13 - shot
      ingredient250 ingredient48 ingredient385 ingredient89 - ingredient
      cocktail459 - cocktail
      dispenser19 dispenser381 dispenser58 dispenser282 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker326)
  (ontable shot2)
  (ontable shot217)
  (ontable shot13)
  (dispenses dispenser19 ingredient250)
  (dispenses dispenser381 ingredient48)
  (dispenses dispenser58 ingredient385)
  (dispenses dispenser282 ingredient89)
  (clean shaker326)
  (clean shot2)
  (clean shot217)
  (clean shot13)
  (empty shaker326)
  (empty shot2)
  (empty shot217)
  (empty shot13)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker326 l0)
  (shaker-level shaker326 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail459 ingredient250)
  (cocktail-part2 cocktail459 ingredient89)
)
 (:goal
  (and
      (contains shot2 cocktail459)
      (contains shot217 cocktail459)
)))
