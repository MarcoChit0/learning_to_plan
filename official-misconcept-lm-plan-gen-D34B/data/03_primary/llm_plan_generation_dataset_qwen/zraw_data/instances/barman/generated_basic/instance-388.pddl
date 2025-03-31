(define (problem prob)
 (:domain barman)
 (:objects 
      shaker464 - shaker
      left right - hand
      shot396 shot419 shot39 - shot
      ingredient238 ingredient448 ingredient187 - ingredient
      cocktail1 - cocktail
      dispenser150 dispenser176 dispenser262 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker464)
  (ontable shot396)
  (ontable shot419)
  (ontable shot39)
  (dispenses dispenser150 ingredient238)
  (dispenses dispenser176 ingredient448)
  (dispenses dispenser262 ingredient187)
  (clean shaker464)
  (clean shot396)
  (clean shot419)
  (clean shot39)
  (empty shaker464)
  (empty shot396)
  (empty shot419)
  (empty shot39)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker464 l0)
  (shaker-level shaker464 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient448)
  (cocktail-part2 cocktail1 ingredient238)
)
 (:goal
  (and
      (contains shot396 cocktail1)
      (contains shot419 cocktail1)
)))
