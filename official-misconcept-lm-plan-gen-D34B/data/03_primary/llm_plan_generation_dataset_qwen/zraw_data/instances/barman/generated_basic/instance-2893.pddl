(define (problem prob)
 (:domain barman)
 (:objects 
      shaker144 - shaker
      left right - hand
      shot363 shot164 shot178 - shot
      ingredient254 ingredient233 - ingredient
      cocktail195 - cocktail
      dispenser73 dispenser217 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker144)
  (ontable shot363)
  (ontable shot164)
  (ontable shot178)
  (dispenses dispenser73 ingredient254)
  (dispenses dispenser217 ingredient233)
  (clean shaker144)
  (clean shot363)
  (clean shot164)
  (clean shot178)
  (empty shaker144)
  (empty shot363)
  (empty shot164)
  (empty shot178)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker144 l0)
  (shaker-level shaker144 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail195 ingredient233)
  (cocktail-part2 cocktail195 ingredient254)
)
 (:goal
  (and
      (contains shot363 cocktail195)
      (contains shot164 ingredient233)
)))
