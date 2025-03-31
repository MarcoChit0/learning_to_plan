(define (problem prob)
 (:domain barman)
 (:objects 
      shaker344 - shaker
      left right - hand
      shot460 shot227 shot80 - shot
      ingredient0 ingredient357 ingredient466 - ingredient
      cocktail495 - cocktail
      dispenser162 dispenser304 dispenser383 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker344)
  (ontable shot460)
  (ontable shot227)
  (ontable shot80)
  (dispenses dispenser162 ingredient0)
  (dispenses dispenser304 ingredient357)
  (dispenses dispenser383 ingredient466)
  (clean shaker344)
  (clean shot460)
  (clean shot227)
  (clean shot80)
  (empty shaker344)
  (empty shot460)
  (empty shot227)
  (empty shot80)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker344 l0)
  (shaker-level shaker344 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail495 ingredient357)
  (cocktail-part2 cocktail495 ingredient466)
)
 (:goal
  (and
      (contains shot460 cocktail495)
      (contains shot227 cocktail495)
)))
