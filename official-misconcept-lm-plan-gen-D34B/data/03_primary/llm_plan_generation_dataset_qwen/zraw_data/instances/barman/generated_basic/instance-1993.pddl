(define (problem prob)
 (:domain barman)
 (:objects 
      shaker368 - shaker
      left right - hand
      shot1 shot0 - shot
      ingredient122 ingredient355 ingredient323 - ingredient
      cocktail29 - cocktail
      dispenser499 dispenser142 dispenser217 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker368)
  (ontable shot1)
  (ontable shot0)
  (dispenses dispenser499 ingredient122)
  (dispenses dispenser142 ingredient355)
  (dispenses dispenser217 ingredient323)
  (clean shaker368)
  (clean shot1)
  (clean shot0)
  (empty shaker368)
  (empty shot1)
  (empty shot0)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker368 l0)
  (shaker-level shaker368 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail29 ingredient122)
  (cocktail-part2 cocktail29 ingredient355)
)
 (:goal
  (and
      (contains shot1 cocktail29)
)))
