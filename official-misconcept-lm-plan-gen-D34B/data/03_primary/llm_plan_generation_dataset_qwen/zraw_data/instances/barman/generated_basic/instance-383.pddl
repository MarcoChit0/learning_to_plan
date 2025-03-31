(define (problem prob)
 (:domain barman)
 (:objects 
      shaker437 - shaker
      left right - hand
      shot243 shot153 shot128 - shot
      ingredient309 ingredient472 ingredient361 - ingredient
      cocktail1 - cocktail
      dispenser423 dispenser389 dispenser342 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker437)
  (ontable shot243)
  (ontable shot153)
  (ontable shot128)
  (dispenses dispenser423 ingredient309)
  (dispenses dispenser389 ingredient472)
  (dispenses dispenser342 ingredient361)
  (clean shaker437)
  (clean shot243)
  (clean shot153)
  (clean shot128)
  (empty shaker437)
  (empty shot243)
  (empty shot153)
  (empty shot128)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker437 l0)
  (shaker-level shaker437 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient361)
  (cocktail-part2 cocktail1 ingredient309)
)
 (:goal
  (and
      (contains shot243 cocktail1)
      (contains shot153 ingredient361)
)))
