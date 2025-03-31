(define (problem prob)
 (:domain barman)
 (:objects 
      shaker136 - shaker
      left right - hand
      shot355 shot220 - shot
      ingredient473 ingredient12 ingredient380 - ingredient
      cocktail287 - cocktail
      dispenser156 dispenser62 dispenser230 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker136)
  (ontable shot355)
  (ontable shot220)
  (dispenses dispenser156 ingredient473)
  (dispenses dispenser62 ingredient12)
  (dispenses dispenser230 ingredient380)
  (clean shaker136)
  (clean shot355)
  (clean shot220)
  (empty shaker136)
  (empty shot355)
  (empty shot220)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker136 l0)
  (shaker-level shaker136 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail287 ingredient473)
  (cocktail-part2 cocktail287 ingredient380)
)
 (:goal
  (and
      (contains shot355 cocktail287)
)))
