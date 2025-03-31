(define (problem prob)
 (:domain barman)
 (:objects 
      shaker19 - shaker
      left right - hand
      shot399 shot218 - shot
      ingredient444 ingredient73 ingredient150 - ingredient
      cocktail190 - cocktail
      dispenser358 dispenser302 dispenser97 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker19)
  (ontable shot399)
  (ontable shot218)
  (dispenses dispenser358 ingredient444)
  (dispenses dispenser302 ingredient73)
  (dispenses dispenser97 ingredient150)
  (clean shaker19)
  (clean shot399)
  (clean shot218)
  (empty shaker19)
  (empty shot399)
  (empty shot218)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker19 l0)
  (shaker-level shaker19 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail190 ingredient73)
  (cocktail-part2 cocktail190 ingredient444)
)
 (:goal
  (and
      (contains shot399 cocktail190)
)))
