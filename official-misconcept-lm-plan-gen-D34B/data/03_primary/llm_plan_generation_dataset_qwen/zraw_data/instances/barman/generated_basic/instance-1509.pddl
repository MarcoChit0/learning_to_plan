(define (problem prob)
 (:domain barman)
 (:objects 
      shaker488 - shaker
      left right - hand
      shot442 shot342 - shot
      ingredient181 ingredient193 ingredient491 ingredient29 - ingredient
      cocktail408 - cocktail
      dispenser135 dispenser109 dispenser290 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker488)
  (ontable shot442)
  (ontable shot342)
  (dispenses dispenser135 ingredient181)
  (dispenses dispenser109 ingredient193)
  (dispenses dispenser290 ingredient491)
  (dispenses dispenser207 ingredient29)
  (clean shaker488)
  (clean shot442)
  (clean shot342)
  (empty shaker488)
  (empty shot442)
  (empty shot342)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker488 l0)
  (shaker-level shaker488 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail408 ingredient29)
  (cocktail-part2 cocktail408 ingredient193)
)
 (:goal
  (and
      (contains shot442 cocktail408)
)))
