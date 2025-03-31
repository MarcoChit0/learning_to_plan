(define (problem prob)
 (:domain barman)
 (:objects 
      shaker441 - shaker
      left right - hand
      shot396 shot359 - shot
      ingredient484 ingredient235 ingredient456 ingredient58 - ingredient
      cocktail15 - cocktail
      dispenser79 dispenser283 dispenser369 dispenser185 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker441)
  (ontable shot396)
  (ontable shot359)
  (dispenses dispenser79 ingredient484)
  (dispenses dispenser283 ingredient235)
  (dispenses dispenser369 ingredient456)
  (dispenses dispenser185 ingredient58)
  (clean shaker441)
  (clean shot396)
  (clean shot359)
  (empty shaker441)
  (empty shot396)
  (empty shot359)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker441 l0)
  (shaker-level shaker441 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail15 ingredient484)
  (cocktail-part2 cocktail15 ingredient456)
)
 (:goal
  (and
      (contains shot396 cocktail15)
)))
