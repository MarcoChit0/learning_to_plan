(define (problem prob)
 (:domain barman)
 (:objects 
      shaker166 - shaker
      left right - hand
      shot28 shot176 - shot
      ingredient117 ingredient399 ingredient106 - ingredient
      cocktail332 - cocktail
      dispenser490 dispenser386 dispenser135 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker166)
  (ontable shot28)
  (ontable shot176)
  (dispenses dispenser490 ingredient117)
  (dispenses dispenser386 ingredient399)
  (dispenses dispenser135 ingredient106)
  (clean shaker166)
  (clean shot28)
  (clean shot176)
  (empty shaker166)
  (empty shot28)
  (empty shot176)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker166 l0)
  (shaker-level shaker166 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail332 ingredient106)
  (cocktail-part2 cocktail332 ingredient399)
)
 (:goal
  (and
      (contains shot28 cocktail332)
)))
