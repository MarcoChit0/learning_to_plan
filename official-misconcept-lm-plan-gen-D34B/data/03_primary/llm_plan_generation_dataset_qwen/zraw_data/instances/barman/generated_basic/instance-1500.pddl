(define (problem prob)
 (:domain barman)
 (:objects 
      shaker322 - shaker
      left right - hand
      shot488 shot325 - shot
      ingredient379 ingredient109 ingredient355 ingredient399 - ingredient
      cocktail180 - cocktail
      dispenser463 dispenser149 dispenser47 dispenser268 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker322)
  (ontable shot488)
  (ontable shot325)
  (dispenses dispenser463 ingredient379)
  (dispenses dispenser149 ingredient109)
  (dispenses dispenser47 ingredient355)
  (dispenses dispenser268 ingredient399)
  (clean shaker322)
  (clean shot488)
  (clean shot325)
  (empty shaker322)
  (empty shot488)
  (empty shot325)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker322 l0)
  (shaker-level shaker322 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail180 ingredient379)
  (cocktail-part2 cocktail180 ingredient399)
)
 (:goal
  (and
      (contains shot488 cocktail180)
)))
