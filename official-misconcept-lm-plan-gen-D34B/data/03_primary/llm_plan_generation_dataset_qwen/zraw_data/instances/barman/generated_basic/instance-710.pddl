(define (problem prob)
 (:domain barman)
 (:objects 
      shaker83 - shaker
      left right - hand
      shot73 shot196 shot282 - shot
      ingredient147 ingredient331 ingredient166 ingredient428 - ingredient
      cocktail1 - cocktail
      dispenser158 dispenser298 dispenser450 dispenser114 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker83)
  (ontable shot73)
  (ontable shot196)
  (ontable shot282)
  (dispenses dispenser158 ingredient147)
  (dispenses dispenser298 ingredient331)
  (dispenses dispenser450 ingredient166)
  (dispenses dispenser114 ingredient428)
  (clean shaker83)
  (clean shot73)
  (clean shot196)
  (clean shot282)
  (empty shaker83)
  (empty shot73)
  (empty shot196)
  (empty shot282)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker83 l0)
  (shaker-level shaker83 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient166)
  (cocktail-part2 cocktail1 ingredient147)
)
 (:goal
  (and
      (contains shot73 cocktail1)
      (contains shot196 cocktail1)
)))
