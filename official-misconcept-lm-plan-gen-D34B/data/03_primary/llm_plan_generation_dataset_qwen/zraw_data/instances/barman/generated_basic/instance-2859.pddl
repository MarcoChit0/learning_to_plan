(define (problem prob)
 (:domain barman)
 (:objects 
      shaker194 - shaker
      left right - hand
      shot304 shot50 - shot
      ingredient152 ingredient83 ingredient57 ingredient179 - ingredient
      cocktail241 - cocktail
      dispenser291 dispenser224 dispenser226 dispenser201 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker194)
  (ontable shot304)
  (ontable shot50)
  (dispenses dispenser291 ingredient152)
  (dispenses dispenser224 ingredient83)
  (dispenses dispenser226 ingredient57)
  (dispenses dispenser201 ingredient179)
  (clean shaker194)
  (clean shot304)
  (clean shot50)
  (empty shaker194)
  (empty shot304)
  (empty shot50)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker194 l0)
  (shaker-level shaker194 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail241 ingredient83)
  (cocktail-part2 cocktail241 ingredient179)
)
 (:goal
  (and
      (contains shot304 cocktail241)
)))
