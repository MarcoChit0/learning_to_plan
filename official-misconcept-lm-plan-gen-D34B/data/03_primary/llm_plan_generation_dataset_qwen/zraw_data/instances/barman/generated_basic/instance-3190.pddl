(define (problem prob)
 (:domain barman)
 (:objects 
      shaker464 - shaker
      left right - hand
      shot262 shot232 - shot
      ingredient24 ingredient263 ingredient74 - ingredient
      cocktail174 - cocktail
      dispenser256 dispenser101 dispenser195 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker464)
  (ontable shot262)
  (ontable shot232)
  (dispenses dispenser256 ingredient24)
  (dispenses dispenser101 ingredient263)
  (dispenses dispenser195 ingredient74)
  (clean shaker464)
  (clean shot262)
  (clean shot232)
  (empty shaker464)
  (empty shot262)
  (empty shot232)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker464 l0)
  (shaker-level shaker464 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail174 ingredient74)
  (cocktail-part2 cocktail174 ingredient263)
)
 (:goal
  (and
      (contains shot262 cocktail174)
)))
