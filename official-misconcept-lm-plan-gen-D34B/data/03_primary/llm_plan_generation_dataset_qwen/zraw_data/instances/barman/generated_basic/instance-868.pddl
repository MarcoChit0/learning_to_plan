(define (problem prob)
 (:domain barman)
 (:objects 
      shaker293 - shaker
      left right - hand
      shot400 shot283 shot328 - shot
      ingredient471 ingredient488 ingredient426 ingredient319 - ingredient
      cocktail1 - cocktail
      dispenser326 dispenser58 dispenser463 dispenser245 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker293)
  (ontable shot400)
  (ontable shot283)
  (ontable shot328)
  (dispenses dispenser326 ingredient471)
  (dispenses dispenser58 ingredient488)
  (dispenses dispenser463 ingredient426)
  (dispenses dispenser245 ingredient319)
  (clean shaker293)
  (clean shot400)
  (clean shot283)
  (clean shot328)
  (empty shaker293)
  (empty shot400)
  (empty shot283)
  (empty shot328)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker293 l0)
  (shaker-level shaker293 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient426)
  (cocktail-part2 cocktail1 ingredient319)
)
 (:goal
  (and
      (contains shot400 cocktail1)
      (contains shot283 cocktail1)
)))
