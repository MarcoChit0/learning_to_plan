(define (problem prob)
 (:domain barman)
 (:objects 
      shaker152 - shaker
      left right - hand
      shot249 shot460 shot399 - shot
      ingredient191 ingredient453 - ingredient
      cocktail84 - cocktail
      dispenser255 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker152)
  (ontable shot249)
  (ontable shot460)
  (ontable shot399)
  (dispenses dispenser255 ingredient191)
  (dispenses dispenser149 ingredient453)
  (clean shaker152)
  (clean shot249)
  (clean shot460)
  (clean shot399)
  (empty shaker152)
  (empty shot249)
  (empty shot460)
  (empty shot399)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker152 l0)
  (shaker-level shaker152 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail84 ingredient453)
  (cocktail-part2 cocktail84 ingredient191)
)
 (:goal
  (and
      (contains shot249 cocktail84)
      (contains shot460 cocktail84)
)))
