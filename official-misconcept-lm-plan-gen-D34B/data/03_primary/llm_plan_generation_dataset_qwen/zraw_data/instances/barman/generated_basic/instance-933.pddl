(define (problem prob)
 (:domain barman)
 (:objects 
      shaker249 - shaker
      left right - hand
      shot112 shot399 shot421 - shot
      ingredient278 ingredient199 ingredient101 ingredient220 - ingredient
      cocktail1 - cocktail
      dispenser239 dispenser124 dispenser411 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker249)
  (ontable shot112)
  (ontable shot399)
  (ontable shot421)
  (dispenses dispenser239 ingredient278)
  (dispenses dispenser124 ingredient199)
  (dispenses dispenser411 ingredient101)
  (dispenses dispenser196 ingredient220)
  (clean shaker249)
  (clean shot112)
  (clean shot399)
  (clean shot421)
  (empty shaker249)
  (empty shot112)
  (empty shot399)
  (empty shot421)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker249 l0)
  (shaker-level shaker249 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient199)
  (cocktail-part2 cocktail1 ingredient101)
)
 (:goal
  (and
      (contains shot112 cocktail1)
      (contains shot399 cocktail1)
)))
