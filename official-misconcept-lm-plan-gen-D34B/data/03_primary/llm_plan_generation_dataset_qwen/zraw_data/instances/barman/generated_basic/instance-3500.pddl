(define (problem prob)
 (:domain barman)
 (:objects 
      shaker444 - shaker
      left right - hand
      shot78 shot141 - shot
      ingredient74 ingredient59 ingredient179 ingredient356 - ingredient
      cocktail0 - cocktail
      dispenser368 dispenser186 dispenser44 dispenser154 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker444)
  (ontable shot78)
  (ontable shot141)
  (dispenses dispenser368 ingredient74)
  (dispenses dispenser186 ingredient59)
  (dispenses dispenser44 ingredient179)
  (dispenses dispenser154 ingredient356)
  (clean shaker444)
  (clean shot78)
  (clean shot141)
  (empty shaker444)
  (empty shot78)
  (empty shot141)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker444 l0)
  (shaker-level shaker444 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail0 ingredient74)
  (cocktail-part2 cocktail0 ingredient59)
)
 (:goal
  (and
      (contains shot78 cocktail0)
)))
