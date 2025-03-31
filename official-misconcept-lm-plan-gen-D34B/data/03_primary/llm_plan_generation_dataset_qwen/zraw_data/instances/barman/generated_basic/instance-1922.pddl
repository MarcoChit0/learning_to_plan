(define (problem prob)
 (:domain barman)
 (:objects 
      shaker277 - shaker
      left right - hand
      shot271 - shot
      ingredient471 ingredient47 ingredient406 ingredient426 - ingredient
      cocktail253 - cocktail
      dispenser178 dispenser468 dispenser472 dispenser164 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker277)
  (ontable shot271)
  (dispenses dispenser178 ingredient471)
  (dispenses dispenser468 ingredient47)
  (dispenses dispenser472 ingredient406)
  (dispenses dispenser164 ingredient426)
  (clean shaker277)
  (clean shot271)
  (empty shaker277)
  (empty shot271)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker277 l0)
  (shaker-level shaker277 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail253 ingredient406)
  (cocktail-part2 cocktail253 ingredient471)
)
 (:goal
  (and
      (contains shot271 cocktail253)
)))
