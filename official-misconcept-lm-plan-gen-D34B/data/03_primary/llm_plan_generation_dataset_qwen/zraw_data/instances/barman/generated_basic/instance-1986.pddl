(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot256 shot161 - shot
      ingredient339 ingredient365 ingredient322 - ingredient
      cocktail119 - cocktail
      dispenser101 dispenser48 dispenser195 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot256)
  (ontable shot161)
  (dispenses dispenser101 ingredient339)
  (dispenses dispenser48 ingredient365)
  (dispenses dispenser195 ingredient322)
  (clean shaker45)
  (clean shot256)
  (clean shot161)
  (empty shaker45)
  (empty shot256)
  (empty shot161)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail119 ingredient339)
  (cocktail-part2 cocktail119 ingredient322)
)
 (:goal
  (and
      (contains shot256 cocktail119)
)))
