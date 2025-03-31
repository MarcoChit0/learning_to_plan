(define (problem prob)
 (:domain barman)
 (:objects 
      shaker273 - shaker
      left right - hand
      shot498 shot41 - shot
      ingredient133 ingredient193 ingredient320 - ingredient
      cocktail1 - cocktail
      dispenser436 dispenser227 dispenser93 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker273)
  (ontable shot498)
  (ontable shot41)
  (dispenses dispenser436 ingredient133)
  (dispenses dispenser227 ingredient193)
  (dispenses dispenser93 ingredient320)
  (clean shaker273)
  (clean shot498)
  (clean shot41)
  (empty shaker273)
  (empty shot498)
  (empty shot41)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker273 l0)
  (shaker-level shaker273 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient133)
  (cocktail-part2 cocktail1 ingredient320)
)
 (:goal
  (and
      (contains shot498 cocktail1)
)))
