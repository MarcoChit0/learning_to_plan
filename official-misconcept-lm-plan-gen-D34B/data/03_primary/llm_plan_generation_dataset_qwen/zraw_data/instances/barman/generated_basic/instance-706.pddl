(define (problem prob)
 (:domain barman)
 (:objects 
      shaker42 - shaker
      left right - hand
      shot131 shot229 shot84 - shot
      ingredient135 ingredient174 ingredient296 ingredient370 - ingredient
      cocktail1 - cocktail
      dispenser184 dispenser11 dispenser220 dispenser431 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker42)
  (ontable shot131)
  (ontable shot229)
  (ontable shot84)
  (dispenses dispenser184 ingredient135)
  (dispenses dispenser11 ingredient174)
  (dispenses dispenser220 ingredient296)
  (dispenses dispenser431 ingredient370)
  (clean shaker42)
  (clean shot131)
  (clean shot229)
  (clean shot84)
  (empty shaker42)
  (empty shot131)
  (empty shot229)
  (empty shot84)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker42 l0)
  (shaker-level shaker42 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient174)
  (cocktail-part2 cocktail1 ingredient296)
)
 (:goal
  (and
      (contains shot131 cocktail1)
      (contains shot229 cocktail1)
)))
