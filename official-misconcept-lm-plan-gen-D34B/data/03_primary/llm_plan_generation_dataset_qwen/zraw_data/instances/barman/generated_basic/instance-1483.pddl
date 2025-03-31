(define (problem prob)
 (:domain barman)
 (:objects 
      shaker277 - shaker
      left right - hand
      shot322 shot376 - shot
      ingredient257 ingredient214 ingredient457 ingredient357 - ingredient
      cocktail228 - cocktail
      dispenser344 dispenser216 dispenser260 dispenser258 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker277)
  (ontable shot322)
  (ontable shot376)
  (dispenses dispenser344 ingredient257)
  (dispenses dispenser216 ingredient214)
  (dispenses dispenser260 ingredient457)
  (dispenses dispenser258 ingredient357)
  (clean shaker277)
  (clean shot322)
  (clean shot376)
  (empty shaker277)
  (empty shot322)
  (empty shot376)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker277 l0)
  (shaker-level shaker277 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail228 ingredient257)
  (cocktail-part2 cocktail228 ingredient214)
)
 (:goal
  (and
      (contains shot322 cocktail228)
)))
