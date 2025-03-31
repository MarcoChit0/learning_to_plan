(define (problem prob)
 (:domain barman)
 (:objects 
      shaker329 - shaker
      left right - hand
      shot335 shot227 shot183 - shot
      ingredient452 ingredient380 - ingredient
      cocktail471 - cocktail
      dispenser275 dispenser72 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker329)
  (ontable shot335)
  (ontable shot227)
  (ontable shot183)
  (dispenses dispenser275 ingredient452)
  (dispenses dispenser72 ingredient380)
  (clean shaker329)
  (clean shot335)
  (clean shot227)
  (clean shot183)
  (empty shaker329)
  (empty shot335)
  (empty shot227)
  (empty shot183)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker329 l0)
  (shaker-level shaker329 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail471 ingredient452)
  (cocktail-part2 cocktail471 ingredient380)
)
 (:goal
  (and
      (contains shot335 cocktail471)
      (contains shot227 ingredient380)
)))
