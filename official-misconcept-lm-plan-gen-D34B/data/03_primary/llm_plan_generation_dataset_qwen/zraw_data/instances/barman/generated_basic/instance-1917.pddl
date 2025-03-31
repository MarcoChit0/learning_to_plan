(define (problem prob)
 (:domain barman)
 (:objects 
      shaker139 - shaker
      left right - hand
      shot290 - shot
      ingredient328 ingredient438 ingredient373 ingredient258 - ingredient
      cocktail448 - cocktail
      dispenser145 dispenser455 dispenser238 dispenser343 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker139)
  (ontable shot290)
  (dispenses dispenser145 ingredient328)
  (dispenses dispenser455 ingredient438)
  (dispenses dispenser238 ingredient373)
  (dispenses dispenser343 ingredient258)
  (clean shaker139)
  (clean shot290)
  (empty shaker139)
  (empty shot290)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker139 l0)
  (shaker-level shaker139 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail448 ingredient258)
  (cocktail-part2 cocktail448 ingredient373)
)
 (:goal
  (and
      (contains shot290 cocktail448)
)))
