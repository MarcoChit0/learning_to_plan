(define (problem prob)
 (:domain barman)
 (:objects 
      shaker457 - shaker
      left right - hand
      shot106 shot73 shot296 - shot
      ingredient198 ingredient160 ingredient199 - ingredient
      cocktail260 - cocktail
      dispenser82 dispenser139 dispenser137 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker457)
  (ontable shot106)
  (ontable shot73)
  (ontable shot296)
  (dispenses dispenser82 ingredient198)
  (dispenses dispenser139 ingredient160)
  (dispenses dispenser137 ingredient199)
  (clean shaker457)
  (clean shot106)
  (clean shot73)
  (clean shot296)
  (empty shaker457)
  (empty shot106)
  (empty shot73)
  (empty shot296)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker457 l0)
  (shaker-level shaker457 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail260 ingredient160)
  (cocktail-part2 cocktail260 ingredient198)
)
 (:goal
  (and
      (contains shot106 cocktail260)
      (contains shot73 ingredient160)
)))
