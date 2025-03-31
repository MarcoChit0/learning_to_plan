(define (problem prob)
 (:domain barman)
 (:objects 
      shaker193 - shaker
      left right - hand
      shot33 shot161 shot125 shot328 - shot
      ingredient322 ingredient131 - ingredient
      cocktail334 - cocktail
      dispenser35 dispenser299 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker193)
  (ontable shot33)
  (ontable shot161)
  (ontable shot125)
  (ontable shot328)
  (dispenses dispenser35 ingredient322)
  (dispenses dispenser299 ingredient131)
  (clean shaker193)
  (clean shot33)
  (clean shot161)
  (clean shot125)
  (clean shot328)
  (empty shaker193)
  (empty shot33)
  (empty shot161)
  (empty shot125)
  (empty shot328)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker193 l0)
  (shaker-level shaker193 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail334 ingredient131)
  (cocktail-part2 cocktail334 ingredient322)
)
 (:goal
  (and
      (contains shot33 cocktail334)
      (contains shot161 cocktail334)
      (contains shot125 ingredient131)
)))
