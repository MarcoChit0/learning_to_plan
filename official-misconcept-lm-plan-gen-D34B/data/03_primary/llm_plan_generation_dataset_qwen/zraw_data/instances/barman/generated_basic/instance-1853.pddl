(define (problem prob)
 (:domain barman)
 (:objects 
      shaker147 - shaker
      left right - hand
      shot322 shot94 - shot
      ingredient306 ingredient340 ingredient178 - ingredient
      cocktail115 - cocktail
      dispenser213 dispenser45 dispenser247 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker147)
  (ontable shot322)
  (ontable shot94)
  (dispenses dispenser213 ingredient306)
  (dispenses dispenser45 ingredient340)
  (dispenses dispenser247 ingredient178)
  (clean shaker147)
  (clean shot322)
  (clean shot94)
  (empty shaker147)
  (empty shot322)
  (empty shot94)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker147 l0)
  (shaker-level shaker147 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail115 ingredient340)
  (cocktail-part2 cocktail115 ingredient178)
)
 (:goal
  (and
      (contains shot322 cocktail115)
)))
