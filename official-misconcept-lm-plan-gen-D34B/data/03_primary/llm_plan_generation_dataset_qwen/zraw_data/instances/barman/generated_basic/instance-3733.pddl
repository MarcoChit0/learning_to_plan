(define (problem prob)
 (:domain barman)
 (:objects 
      shaker203 - shaker
      left right - hand
      shot403 shot427 shot97 - shot
      ingredient209 ingredient227 ingredient199 - ingredient
      cocktail297 - cocktail
      dispenser372 dispenser492 dispenser375 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker203)
  (ontable shot403)
  (ontable shot427)
  (ontable shot97)
  (dispenses dispenser372 ingredient209)
  (dispenses dispenser492 ingredient227)
  (dispenses dispenser375 ingredient199)
  (clean shaker203)
  (clean shot403)
  (clean shot427)
  (clean shot97)
  (empty shaker203)
  (empty shot403)
  (empty shot427)
  (empty shot97)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker203 l0)
  (shaker-level shaker203 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail297 ingredient199)
  (cocktail-part2 cocktail297 ingredient227)
)
 (:goal
  (and
      (contains shot403 cocktail297)
      (contains shot427 ingredient199)
)))
