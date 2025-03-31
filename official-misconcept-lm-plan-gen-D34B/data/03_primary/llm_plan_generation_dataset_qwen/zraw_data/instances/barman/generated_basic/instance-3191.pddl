(define (problem prob)
 (:domain barman)
 (:objects 
      shaker197 - shaker
      left right - hand
      shot395 shot227 - shot
      ingredient489 ingredient104 ingredient434 - ingredient
      cocktail143 - cocktail
      dispenser426 dispenser480 dispenser467 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker197)
  (ontable shot395)
  (ontable shot227)
  (dispenses dispenser426 ingredient489)
  (dispenses dispenser480 ingredient104)
  (dispenses dispenser467 ingredient434)
  (clean shaker197)
  (clean shot395)
  (clean shot227)
  (empty shaker197)
  (empty shot395)
  (empty shot227)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker197 l0)
  (shaker-level shaker197 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail143 ingredient104)
  (cocktail-part2 cocktail143 ingredient434)
)
 (:goal
  (and
      (contains shot395 cocktail143)
)))
