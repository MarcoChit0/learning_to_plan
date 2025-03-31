(define (problem prob)
 (:domain barman)
 (:objects 
      shaker124 - shaker
      left right - hand
      shot238 shot441 - shot
      ingredient178 ingredient404 ingredient217 - ingredient
      cocktail489 - cocktail
      dispenser287 dispenser49 dispenser345 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker124)
  (ontable shot238)
  (ontable shot441)
  (dispenses dispenser287 ingredient178)
  (dispenses dispenser49 ingredient404)
  (dispenses dispenser345 ingredient217)
  (clean shaker124)
  (clean shot238)
  (clean shot441)
  (empty shaker124)
  (empty shot238)
  (empty shot441)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker124 l0)
  (shaker-level shaker124 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail489 ingredient217)
  (cocktail-part2 cocktail489 ingredient178)
)
 (:goal
  (and
      (contains shot238 cocktail489)
)))
