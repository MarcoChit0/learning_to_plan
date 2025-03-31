(define (problem prob)
 (:domain barman)
 (:objects 
      shaker418 - shaker
      left right - hand
      shot416 shot252 shot389 - shot
      ingredient94 ingredient452 - ingredient
      cocktail431 - cocktail
      dispenser259 dispenser463 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker418)
  (ontable shot416)
  (ontable shot252)
  (ontable shot389)
  (dispenses dispenser259 ingredient94)
  (dispenses dispenser463 ingredient452)
  (clean shaker418)
  (clean shot416)
  (clean shot252)
  (clean shot389)
  (empty shaker418)
  (empty shot416)
  (empty shot252)
  (empty shot389)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker418 l0)
  (shaker-level shaker418 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail431 ingredient452)
  (cocktail-part2 cocktail431 ingredient94)
)
 (:goal
  (and
      (contains shot416 cocktail431)
      (contains shot252 ingredient452)
)))
