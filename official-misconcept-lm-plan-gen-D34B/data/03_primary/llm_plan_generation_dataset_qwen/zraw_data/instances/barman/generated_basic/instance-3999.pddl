(define (problem prob)
 (:domain barman)
 (:objects 
      shaker249 - shaker
      left right - hand
      shot376 shot149 shot44 - shot
      ingredient152 ingredient249 - ingredient
      cocktail85 - cocktail
      dispenser36 dispenser259 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker249)
  (ontable shot376)
  (ontable shot149)
  (ontable shot44)
  (dispenses dispenser36 ingredient152)
  (dispenses dispenser259 ingredient249)
  (clean shaker249)
  (clean shot376)
  (clean shot149)
  (clean shot44)
  (empty shaker249)
  (empty shot376)
  (empty shot149)
  (empty shot44)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker249 l0)
  (shaker-level shaker249 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail85 ingredient152)
  (cocktail-part2 cocktail85 ingredient249)
)
 (:goal
  (and
      (contains shot376 cocktail85)
      (contains shot149 ingredient249)
)))
