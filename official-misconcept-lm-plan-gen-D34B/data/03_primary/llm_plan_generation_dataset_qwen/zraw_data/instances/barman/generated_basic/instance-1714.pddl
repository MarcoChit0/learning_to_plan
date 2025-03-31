(define (problem prob)
 (:domain barman)
 (:objects 
      shaker114 - shaker
      left right - hand
      shot292 shot67 - shot
      ingredient441 ingredient187 ingredient46 ingredient421 - ingredient
      cocktail405 - cocktail
      dispenser226 dispenser205 dispenser367 dispenser16 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker114)
  (ontable shot292)
  (ontable shot67)
  (dispenses dispenser226 ingredient441)
  (dispenses dispenser205 ingredient187)
  (dispenses dispenser367 ingredient46)
  (dispenses dispenser16 ingredient421)
  (clean shaker114)
  (clean shot292)
  (clean shot67)
  (empty shaker114)
  (empty shot292)
  (empty shot67)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker114 l0)
  (shaker-level shaker114 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail405 ingredient46)
  (cocktail-part2 cocktail405 ingredient187)
)
 (:goal
  (and
      (contains shot292 cocktail405)
)))
