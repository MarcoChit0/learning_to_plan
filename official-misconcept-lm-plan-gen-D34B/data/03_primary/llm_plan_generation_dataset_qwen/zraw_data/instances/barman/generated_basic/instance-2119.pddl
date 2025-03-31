(define (problem prob)
 (:domain barman)
 (:objects 
      shaker104 - shaker
      left right - hand
      shot31 shot221 shot122 - shot
      ingredient247 ingredient120 - ingredient
      cocktail279 - cocktail
      dispenser115 dispenser315 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker104)
  (ontable shot31)
  (ontable shot221)
  (ontable shot122)
  (dispenses dispenser115 ingredient247)
  (dispenses dispenser315 ingredient120)
  (clean shaker104)
  (clean shot31)
  (clean shot221)
  (clean shot122)
  (empty shaker104)
  (empty shot31)
  (empty shot221)
  (empty shot122)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker104 l0)
  (shaker-level shaker104 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail279 ingredient247)
  (cocktail-part2 cocktail279 ingredient120)
)
 (:goal
  (and
      (contains shot31 cocktail279)
      (contains shot221 ingredient247)
)))
