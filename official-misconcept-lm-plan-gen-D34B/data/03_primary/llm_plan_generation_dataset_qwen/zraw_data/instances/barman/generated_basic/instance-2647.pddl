(define (problem prob)
 (:domain barman)
 (:objects 
      shaker48 - shaker
      left right - hand
      shot48 shot493 - shot
      ingredient355 ingredient368 ingredient404 - ingredient
      cocktail93 - cocktail
      dispenser459 dispenser306 dispenser5 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker48)
  (ontable shot48)
  (ontable shot493)
  (dispenses dispenser459 ingredient355)
  (dispenses dispenser306 ingredient368)
  (dispenses dispenser5 ingredient404)
  (clean shaker48)
  (clean shot48)
  (clean shot493)
  (empty shaker48)
  (empty shot48)
  (empty shot493)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker48 l0)
  (shaker-level shaker48 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail93 ingredient368)
  (cocktail-part2 cocktail93 ingredient355)
)
 (:goal
  (and
      (contains shot48 cocktail93)
)))
