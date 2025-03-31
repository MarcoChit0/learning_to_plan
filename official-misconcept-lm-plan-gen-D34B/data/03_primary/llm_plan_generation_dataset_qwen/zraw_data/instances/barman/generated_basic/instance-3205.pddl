(define (problem prob)
 (:domain barman)
 (:objects 
      shaker292 - shaker
      left right - hand
      shot143 shot304 - shot
      ingredient291 ingredient329 ingredient439 - ingredient
      cocktail359 - cocktail
      dispenser109 dispenser338 dispenser331 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker292)
  (ontable shot143)
  (ontable shot304)
  (dispenses dispenser109 ingredient291)
  (dispenses dispenser338 ingredient329)
  (dispenses dispenser331 ingredient439)
  (clean shaker292)
  (clean shot143)
  (clean shot304)
  (empty shaker292)
  (empty shot143)
  (empty shot304)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker292 l0)
  (shaker-level shaker292 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail359 ingredient439)
  (cocktail-part2 cocktail359 ingredient291)
)
 (:goal
  (and
      (contains shot143 cocktail359)
)))
