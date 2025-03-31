(define (problem prob)
 (:domain barman)
 (:objects 
      shaker467 - shaker
      left right - hand
      shot462 shot232 - shot
      ingredient40 ingredient241 ingredient59 ingredient146 - ingredient
      cocktail243 - cocktail
      dispenser422 dispenser338 dispenser173 dispenser167 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker467)
  (ontable shot462)
  (ontable shot232)
  (dispenses dispenser422 ingredient40)
  (dispenses dispenser338 ingredient241)
  (dispenses dispenser173 ingredient59)
  (dispenses dispenser167 ingredient146)
  (clean shaker467)
  (clean shot462)
  (clean shot232)
  (empty shaker467)
  (empty shot462)
  (empty shot232)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker467 l0)
  (shaker-level shaker467 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail243 ingredient40)
  (cocktail-part2 cocktail243 ingredient146)
)
 (:goal
  (and
      (contains shot462 cocktail243)
)))
