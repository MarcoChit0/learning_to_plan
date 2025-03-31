(define (problem prob)
 (:domain barman)
 (:objects 
      shaker141 - shaker
      left right - hand
      shot402 shot466 - shot
      ingredient96 ingredient292 - ingredient
      cocktail142 - cocktail
      dispenser399 dispenser465 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker141)
  (ontable shot402)
  (ontable shot466)
  (dispenses dispenser399 ingredient96)
  (dispenses dispenser465 ingredient292)
  (clean shaker141)
  (clean shot402)
  (clean shot466)
  (empty shaker141)
  (empty shot402)
  (empty shot466)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker141 l0)
  (shaker-level shaker141 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail142 ingredient292)
  (cocktail-part2 cocktail142 ingredient96)
)
 (:goal
  (and
      (contains shot402 cocktail142)
)))
