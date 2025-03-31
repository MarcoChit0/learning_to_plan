(define (problem prob)
 (:domain barman)
 (:objects 
      shaker127 - shaker
      left right - hand
      shot314 shot163 shot132 - shot
      ingredient61 ingredient115 ingredient336 - ingredient
      cocktail290 - cocktail
      dispenser236 dispenser41 dispenser486 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker127)
  (ontable shot314)
  (ontable shot163)
  (ontable shot132)
  (dispenses dispenser236 ingredient61)
  (dispenses dispenser41 ingredient115)
  (dispenses dispenser486 ingredient336)
  (clean shaker127)
  (clean shot314)
  (clean shot163)
  (clean shot132)
  (empty shaker127)
  (empty shot314)
  (empty shot163)
  (empty shot132)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker127 l0)
  (shaker-level shaker127 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail290 ingredient336)
  (cocktail-part2 cocktail290 ingredient61)
)
 (:goal
  (and
      (contains shot314 cocktail290)
      (contains shot163 cocktail290)
)))
