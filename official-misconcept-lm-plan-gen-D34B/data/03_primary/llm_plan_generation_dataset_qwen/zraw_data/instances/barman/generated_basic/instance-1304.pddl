(define (problem prob)
 (:domain barman)
 (:objects 
      shaker39 - shaker
      left right - hand
      shot399 shot290 shot129 - shot
      ingredient144 ingredient88 - ingredient
      cocktail48 - cocktail
      dispenser200 dispenser137 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker39)
  (ontable shot399)
  (ontable shot290)
  (ontable shot129)
  (dispenses dispenser200 ingredient144)
  (dispenses dispenser137 ingredient88)
  (clean shaker39)
  (clean shot399)
  (clean shot290)
  (clean shot129)
  (empty shaker39)
  (empty shot399)
  (empty shot290)
  (empty shot129)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker39 l0)
  (shaker-level shaker39 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail48 ingredient88)
  (cocktail-part2 cocktail48 ingredient144)
)
 (:goal
  (and
      (contains shot399 cocktail48)
      (contains shot290 cocktail48)
)))
