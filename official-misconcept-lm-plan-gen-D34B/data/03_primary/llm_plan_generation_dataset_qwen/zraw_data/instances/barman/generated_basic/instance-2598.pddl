(define (problem prob)
 (:domain barman)
 (:objects 
      shaker418 - shaker
      left right - hand
      shot400 shot188 - shot
      ingredient261 ingredient20 ingredient356 - ingredient
      cocktail422 - cocktail
      dispenser156 dispenser259 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker418)
  (ontable shot400)
  (ontable shot188)
  (dispenses dispenser156 ingredient261)
  (dispenses dispenser259 ingredient20)
  (dispenses dispenser214 ingredient356)
  (clean shaker418)
  (clean shot400)
  (clean shot188)
  (empty shaker418)
  (empty shot400)
  (empty shot188)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker418 l0)
  (shaker-level shaker418 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail422 ingredient261)
  (cocktail-part2 cocktail422 ingredient20)
)
 (:goal
  (and
      (contains shot400 cocktail422)
)))
