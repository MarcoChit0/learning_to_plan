(define (problem prob)
 (:domain barman)
 (:objects 
      shaker64 - shaker
      left right - hand
      shot82 shot240 shot299 - shot
      ingredient218 ingredient254 ingredient198 ingredient184 - ingredient
      cocktail1 - cocktail
      dispenser13 dispenser479 dispenser268 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker64)
  (ontable shot82)
  (ontable shot240)
  (ontable shot299)
  (dispenses dispenser13 ingredient218)
  (dispenses dispenser479 ingredient254)
  (dispenses dispenser268 ingredient198)
  (dispenses dispenser397 ingredient184)
  (clean shaker64)
  (clean shot82)
  (clean shot240)
  (clean shot299)
  (empty shaker64)
  (empty shot82)
  (empty shot240)
  (empty shot299)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker64 l0)
  (shaker-level shaker64 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient254)
  (cocktail-part2 cocktail1 ingredient218)
)
 (:goal
  (and
      (contains shot82 cocktail1)
      (contains shot240 cocktail1)
)))
