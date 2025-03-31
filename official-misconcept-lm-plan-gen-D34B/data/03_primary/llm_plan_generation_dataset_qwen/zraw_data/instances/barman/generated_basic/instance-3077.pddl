(define (problem prob)
 (:domain barman)
 (:objects 
      shaker128 - shaker
      left right - hand
      shot399 shot195 - shot
      ingredient321 ingredient255 - ingredient
      cocktail188 - cocktail
      dispenser106 dispenser38 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker128)
  (ontable shot399)
  (ontable shot195)
  (dispenses dispenser106 ingredient321)
  (dispenses dispenser38 ingredient255)
  (clean shaker128)
  (clean shot399)
  (clean shot195)
  (empty shaker128)
  (empty shot399)
  (empty shot195)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker128 l0)
  (shaker-level shaker128 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail188 ingredient321)
  (cocktail-part2 cocktail188 ingredient255)
)
 (:goal
  (and
      (contains shot399 cocktail188)
)))
