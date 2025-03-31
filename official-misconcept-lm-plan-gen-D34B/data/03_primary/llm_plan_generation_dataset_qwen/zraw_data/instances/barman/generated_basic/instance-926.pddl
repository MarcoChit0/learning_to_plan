(define (problem prob)
 (:domain barman)
 (:objects 
      shaker262 - shaker
      left right - hand
      shot418 shot270 - shot
      ingredient459 ingredient234 ingredient492 - ingredient
      cocktail1 - cocktail
      dispenser220 dispenser322 dispenser198 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker262)
  (ontable shot418)
  (ontable shot270)
  (dispenses dispenser220 ingredient459)
  (dispenses dispenser322 ingredient234)
  (dispenses dispenser198 ingredient492)
  (clean shaker262)
  (clean shot418)
  (clean shot270)
  (empty shaker262)
  (empty shot418)
  (empty shot270)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker262 l0)
  (shaker-level shaker262 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient234)
  (cocktail-part2 cocktail1 ingredient459)
)
 (:goal
  (and
      (contains shot418 cocktail1)
)))
