(define (problem prob)
 (:domain barman)
 (:objects 
      shaker427 - shaker
      left right - hand
      shot178 shot267 - shot
      ingredient224 ingredient253 ingredient316 - ingredient
      cocktail265 - cocktail
      dispenser353 dispenser266 dispenser78 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker427)
  (ontable shot178)
  (ontable shot267)
  (dispenses dispenser353 ingredient224)
  (dispenses dispenser266 ingredient253)
  (dispenses dispenser78 ingredient316)
  (clean shaker427)
  (clean shot178)
  (clean shot267)
  (empty shaker427)
  (empty shot178)
  (empty shot267)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker427 l0)
  (shaker-level shaker427 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail265 ingredient316)
  (cocktail-part2 cocktail265 ingredient224)
)
 (:goal
  (and
      (contains shot178 cocktail265)
)))
