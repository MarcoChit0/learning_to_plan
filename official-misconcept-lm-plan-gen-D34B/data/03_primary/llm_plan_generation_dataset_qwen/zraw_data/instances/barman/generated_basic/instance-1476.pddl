(define (problem prob)
 (:domain barman)
 (:objects 
      shaker390 - shaker
      left right - hand
      shot183 shot293 - shot
      ingredient157 ingredient123 ingredient41 ingredient312 - ingredient
      cocktail119 - cocktail
      dispenser289 dispenser247 dispenser171 dispenser406 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker390)
  (ontable shot183)
  (ontable shot293)
  (dispenses dispenser289 ingredient157)
  (dispenses dispenser247 ingredient123)
  (dispenses dispenser171 ingredient41)
  (dispenses dispenser406 ingredient312)
  (clean shaker390)
  (clean shot183)
  (clean shot293)
  (empty shaker390)
  (empty shot183)
  (empty shot293)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker390 l0)
  (shaker-level shaker390 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail119 ingredient157)
  (cocktail-part2 cocktail119 ingredient41)
)
 (:goal
  (and
      (contains shot183 cocktail119)
)))
