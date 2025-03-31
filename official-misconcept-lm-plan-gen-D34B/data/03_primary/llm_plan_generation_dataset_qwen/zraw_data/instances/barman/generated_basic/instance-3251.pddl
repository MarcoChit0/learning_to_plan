(define (problem prob)
 (:domain barman)
 (:objects 
      shaker56 - shaker
      left right - hand
      shot315 shot283 shot0 - shot
      ingredient166 ingredient269 ingredient152 ingredient356 - ingredient
      cocktail170 - cocktail
      dispenser118 dispenser22 dispenser440 dispenser4 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker56)
  (ontable shot315)
  (ontable shot283)
  (ontable shot0)
  (dispenses dispenser118 ingredient166)
  (dispenses dispenser22 ingredient269)
  (dispenses dispenser440 ingredient152)
  (dispenses dispenser4 ingredient356)
  (clean shaker56)
  (clean shot315)
  (clean shot283)
  (clean shot0)
  (empty shaker56)
  (empty shot315)
  (empty shot283)
  (empty shot0)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker56 l0)
  (shaker-level shaker56 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail170 ingredient152)
  (cocktail-part2 cocktail170 ingredient166)
)
 (:goal
  (and
      (contains shot315 cocktail170)
      (contains shot283 cocktail170)
)))
