(define (problem prob)
 (:domain barman)
 (:objects 
      shaker407 - shaker
      left right - hand
      shot227 - shot
      ingredient389 ingredient184 ingredient228 - ingredient
      cocktail328 - cocktail
      dispenser39 dispenser75 dispenser441 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker407)
  (ontable shot227)
  (dispenses dispenser39 ingredient389)
  (dispenses dispenser75 ingredient184)
  (dispenses dispenser441 ingredient228)
  (clean shaker407)
  (clean shot227)
  (empty shaker407)
  (empty shot227)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker407 l0)
  (shaker-level shaker407 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail328 ingredient389)
  (cocktail-part2 cocktail328 ingredient228)
)
 (:goal
  (and
      (contains shot227 cocktail328)
)))
