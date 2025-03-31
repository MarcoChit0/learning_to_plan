(define (problem prob)
 (:domain barman)
 (:objects 
      shaker249 - shaker
      left right - hand
      shot417 shot321 shot359 - shot
      ingredient366 ingredient44 ingredient155 - ingredient
      cocktail32 - cocktail
      dispenser298 dispenser448 dispenser25 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker249)
  (ontable shot417)
  (ontable shot321)
  (ontable shot359)
  (dispenses dispenser298 ingredient366)
  (dispenses dispenser448 ingredient44)
  (dispenses dispenser25 ingredient155)
  (clean shaker249)
  (clean shot417)
  (clean shot321)
  (clean shot359)
  (empty shaker249)
  (empty shot417)
  (empty shot321)
  (empty shot359)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker249 l0)
  (shaker-level shaker249 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail32 ingredient366)
  (cocktail-part2 cocktail32 ingredient44)
)
 (:goal
  (and
      (contains shot417 cocktail32)
      (contains shot321 cocktail32)
)))
