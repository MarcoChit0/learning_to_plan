(define (problem prob)
 (:domain barman)
 (:objects 
      shaker398 - shaker
      left right - hand
      shot349 shot440 shot399 - shot
      ingredient423 ingredient99 ingredient9 - ingredient
      cocktail451 - cocktail
      dispenser259 dispenser28 dispenser421 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker398)
  (ontable shot349)
  (ontable shot440)
  (ontable shot399)
  (dispenses dispenser259 ingredient423)
  (dispenses dispenser28 ingredient99)
  (dispenses dispenser421 ingredient9)
  (clean shaker398)
  (clean shot349)
  (clean shot440)
  (clean shot399)
  (empty shaker398)
  (empty shot349)
  (empty shot440)
  (empty shot399)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker398 l0)
  (shaker-level shaker398 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail451 ingredient423)
  (cocktail-part2 cocktail451 ingredient99)
)
 (:goal
  (and
      (contains shot349 cocktail451)
      (contains shot440 cocktail451)
)))
