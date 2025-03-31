(define (problem prob)
 (:domain barman)
 (:objects 
      shaker498 - shaker
      left right - hand
      shot138 shot87 - shot
      ingredient249 ingredient192 ingredient295 - ingredient
      cocktail328 - cocktail
      dispenser187 dispenser329 dispenser228 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker498)
  (ontable shot138)
  (ontable shot87)
  (dispenses dispenser187 ingredient249)
  (dispenses dispenser329 ingredient192)
  (dispenses dispenser228 ingredient295)
  (clean shaker498)
  (clean shot138)
  (clean shot87)
  (empty shaker498)
  (empty shot138)
  (empty shot87)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker498 l0)
  (shaker-level shaker498 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail328 ingredient249)
  (cocktail-part2 cocktail328 ingredient295)
)
 (:goal
  (and
      (contains shot138 cocktail328)
)))
