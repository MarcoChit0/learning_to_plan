(define (problem prob)
 (:domain barman)
 (:objects 
      shaker65 - shaker
      left right - hand
      shot3 shot480 - shot
      ingredient454 ingredient389 ingredient420 ingredient20 - ingredient
      cocktail48 - cocktail
      dispenser453 dispenser257 dispenser423 dispenser392 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker65)
  (ontable shot3)
  (ontable shot480)
  (dispenses dispenser453 ingredient454)
  (dispenses dispenser257 ingredient389)
  (dispenses dispenser423 ingredient420)
  (dispenses dispenser392 ingredient20)
  (clean shaker65)
  (clean shot3)
  (clean shot480)
  (empty shaker65)
  (empty shot3)
  (empty shot480)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker65 l0)
  (shaker-level shaker65 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail48 ingredient20)
  (cocktail-part2 cocktail48 ingredient389)
)
 (:goal
  (and
      (contains shot3 cocktail48)
)))
