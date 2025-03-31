(define (problem prob)
 (:domain barman)
 (:objects 
      shaker323 - shaker
      left right - hand
      shot271 shot290 - shot
      ingredient207 ingredient123 ingredient249 ingredient190 - ingredient
      cocktail1 - cocktail
      dispenser256 dispenser310 dispenser190 dispenser296 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker323)
  (ontable shot271)
  (ontable shot290)
  (dispenses dispenser256 ingredient207)
  (dispenses dispenser310 ingredient123)
  (dispenses dispenser190 ingredient249)
  (dispenses dispenser296 ingredient190)
  (clean shaker323)
  (clean shot271)
  (clean shot290)
  (empty shaker323)
  (empty shot271)
  (empty shot290)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker323 l0)
  (shaker-level shaker323 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient249)
  (cocktail-part2 cocktail1 ingredient123)
)
 (:goal
  (and
      (contains shot271 cocktail1)
)))
