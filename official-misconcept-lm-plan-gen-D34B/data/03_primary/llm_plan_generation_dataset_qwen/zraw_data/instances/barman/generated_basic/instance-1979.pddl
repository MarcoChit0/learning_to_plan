(define (problem prob)
 (:domain barman)
 (:objects 
      shaker2 - shaker
      left right - hand
      shot215 shot139 - shot
      ingredient232 ingredient50 ingredient483 - ingredient
      cocktail156 - cocktail
      dispenser364 dispenser465 dispenser46 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker2)
  (ontable shot215)
  (ontable shot139)
  (dispenses dispenser364 ingredient232)
  (dispenses dispenser465 ingredient50)
  (dispenses dispenser46 ingredient483)
  (clean shaker2)
  (clean shot215)
  (clean shot139)
  (empty shaker2)
  (empty shot215)
  (empty shot139)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker2 l0)
  (shaker-level shaker2 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail156 ingredient232)
  (cocktail-part2 cocktail156 ingredient50)
)
 (:goal
  (and
      (contains shot215 cocktail156)
)))
