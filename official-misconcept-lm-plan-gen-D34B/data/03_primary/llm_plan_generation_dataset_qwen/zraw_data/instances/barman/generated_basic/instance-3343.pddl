(define (problem prob)
 (:domain barman)
 (:objects 
      shaker436 - shaker
      left right - hand
      shot18 shot479 - shot
      ingredient111 ingredient307 - ingredient
      cocktail281 - cocktail
      dispenser260 dispenser343 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker436)
  (ontable shot18)
  (ontable shot479)
  (dispenses dispenser260 ingredient111)
  (dispenses dispenser343 ingredient307)
  (clean shaker436)
  (clean shot18)
  (clean shot479)
  (empty shaker436)
  (empty shot18)
  (empty shot479)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker436 l0)
  (shaker-level shaker436 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail281 ingredient307)
  (cocktail-part2 cocktail281 ingredient111)
)
 (:goal
  (and
      (contains shot18 cocktail281)
)))
