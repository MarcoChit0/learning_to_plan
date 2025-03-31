(define (problem prob)
 (:domain barman)
 (:objects 
      shaker322 - shaker
      left right - hand
      shot235 shot105 - shot
      ingredient39 ingredient370 - ingredient
      cocktail1 - cocktail
      dispenser354 dispenser115 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker322)
  (ontable shot235)
  (ontable shot105)
  (dispenses dispenser354 ingredient39)
  (dispenses dispenser115 ingredient370)
  (clean shaker322)
  (clean shot235)
  (clean shot105)
  (empty shaker322)
  (empty shot235)
  (empty shot105)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker322 l0)
  (shaker-level shaker322 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient370)
  (cocktail-part2 cocktail1 ingredient39)
)
 (:goal
  (and
      (contains shot235 cocktail1)
)))
