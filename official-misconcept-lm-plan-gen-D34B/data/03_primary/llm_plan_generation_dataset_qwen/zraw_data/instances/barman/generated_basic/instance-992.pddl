(define (problem prob)
 (:domain barman)
 (:objects 
      shaker244 - shaker
      left right - hand
      shot136 shot345 - shot
      ingredient277 ingredient227 - ingredient
      cocktail1 - cocktail
      dispenser218 dispenser479 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker244)
  (ontable shot136)
  (ontable shot345)
  (dispenses dispenser218 ingredient277)
  (dispenses dispenser479 ingredient227)
  (clean shaker244)
  (clean shot136)
  (clean shot345)
  (empty shaker244)
  (empty shot136)
  (empty shot345)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker244 l0)
  (shaker-level shaker244 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient227)
  (cocktail-part2 cocktail1 ingredient277)
)
 (:goal
  (and
      (contains shot136 cocktail1)
)))
