(define (problem prob)
 (:domain barman)
 (:objects 
      shaker214 - shaker
      left right - hand
      shot221 - shot
      ingredient6 ingredient362 ingredient240 - ingredient
      cocktail1 - cocktail
      dispenser245 dispenser86 dispenser210 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker214)
  (ontable shot221)
  (dispenses dispenser245 ingredient6)
  (dispenses dispenser86 ingredient362)
  (dispenses dispenser210 ingredient240)
  (clean shaker214)
  (clean shot221)
  (empty shaker214)
  (empty shot221)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker214 l0)
  (shaker-level shaker214 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient240)
  (cocktail-part2 cocktail1 ingredient6)
)
 (:goal
  (and
      (contains shot221 cocktail1)
)))
