(define (problem prob)
 (:domain barman)
 (:objects 
      shaker52 - shaker
      left right - hand
      shot354 - shot
      ingredient234 ingredient28 ingredient38 ingredient449 - ingredient
      cocktail255 - cocktail
      dispenser146 dispenser88 dispenser388 dispenser422 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker52)
  (ontable shot354)
  (dispenses dispenser146 ingredient234)
  (dispenses dispenser88 ingredient28)
  (dispenses dispenser388 ingredient38)
  (dispenses dispenser422 ingredient449)
  (clean shaker52)
  (clean shot354)
  (empty shaker52)
  (empty shot354)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker52 l0)
  (shaker-level shaker52 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail255 ingredient449)
  (cocktail-part2 cocktail255 ingredient28)
)
 (:goal
  (and
      (contains shot354 cocktail255)
)))
