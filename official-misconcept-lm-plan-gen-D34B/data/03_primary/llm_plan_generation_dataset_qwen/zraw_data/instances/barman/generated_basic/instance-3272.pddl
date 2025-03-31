(define (problem prob)
 (:domain barman)
 (:objects 
      shaker404 - shaker
      left right - hand
      shot330 shot450 - shot
      ingredient247 ingredient144 ingredient113 - ingredient
      cocktail342 - cocktail
      dispenser277 dispenser268 dispenser326 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker404)
  (ontable shot330)
  (ontable shot450)
  (dispenses dispenser277 ingredient247)
  (dispenses dispenser268 ingredient144)
  (dispenses dispenser326 ingredient113)
  (clean shaker404)
  (clean shot330)
  (clean shot450)
  (empty shaker404)
  (empty shot330)
  (empty shot450)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker404 l0)
  (shaker-level shaker404 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail342 ingredient247)
  (cocktail-part2 cocktail342 ingredient113)
)
 (:goal
  (and
      (contains shot330 cocktail342)
)))
