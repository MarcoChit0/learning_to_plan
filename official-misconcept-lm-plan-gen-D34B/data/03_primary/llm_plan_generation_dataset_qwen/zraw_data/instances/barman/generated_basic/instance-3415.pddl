(define (problem prob)
 (:domain barman)
 (:objects 
      shaker4 - shaker
      left right - hand
      shot478 shot202 - shot
      ingredient276 ingredient58 ingredient388 - ingredient
      cocktail114 - cocktail
      dispenser212 dispenser149 dispenser48 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker4)
  (ontable shot478)
  (ontable shot202)
  (dispenses dispenser212 ingredient276)
  (dispenses dispenser149 ingredient58)
  (dispenses dispenser48 ingredient388)
  (clean shaker4)
  (clean shot478)
  (clean shot202)
  (empty shaker4)
  (empty shot478)
  (empty shot202)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker4 l0)
  (shaker-level shaker4 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail114 ingredient276)
  (cocktail-part2 cocktail114 ingredient388)
)
 (:goal
  (and
      (contains shot478 cocktail114)
)))
