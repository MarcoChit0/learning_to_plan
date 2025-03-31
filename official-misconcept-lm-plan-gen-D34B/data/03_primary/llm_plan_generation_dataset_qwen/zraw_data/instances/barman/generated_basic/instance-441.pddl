(define (problem prob)
 (:domain barman)
 (:objects 
      shaker56 - shaker
      left right - hand
      shot266 shot365 - shot
      ingredient493 ingredient216 ingredient444 - ingredient
      cocktail1 - cocktail
      dispenser381 dispenser136 dispenser68 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker56)
  (ontable shot266)
  (ontable shot365)
  (dispenses dispenser381 ingredient493)
  (dispenses dispenser136 ingredient216)
  (dispenses dispenser68 ingredient444)
  (clean shaker56)
  (clean shot266)
  (clean shot365)
  (empty shaker56)
  (empty shot266)
  (empty shot365)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker56 l0)
  (shaker-level shaker56 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient216)
  (cocktail-part2 cocktail1 ingredient444)
)
 (:goal
  (and
      (contains shot266 cocktail1)
)))
