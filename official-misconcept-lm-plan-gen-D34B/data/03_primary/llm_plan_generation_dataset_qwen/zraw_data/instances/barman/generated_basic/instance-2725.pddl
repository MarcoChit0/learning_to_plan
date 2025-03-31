(define (problem prob)
 (:domain barman)
 (:objects 
      shaker303 - shaker
      left right - hand
      shot317 shot36 shot249 - shot
      ingredient141 ingredient98 ingredient467 - ingredient
      cocktail13 - cocktail
      dispenser92 dispenser131 dispenser390 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker303)
  (ontable shot317)
  (ontable shot36)
  (ontable shot249)
  (dispenses dispenser92 ingredient141)
  (dispenses dispenser131 ingredient98)
  (dispenses dispenser390 ingredient467)
  (clean shaker303)
  (clean shot317)
  (clean shot36)
  (clean shot249)
  (empty shaker303)
  (empty shot317)
  (empty shot36)
  (empty shot249)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker303 l0)
  (shaker-level shaker303 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail13 ingredient141)
  (cocktail-part2 cocktail13 ingredient467)
)
 (:goal
  (and
      (contains shot317 cocktail13)
      (contains shot36 ingredient467)
)))
