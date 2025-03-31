(define (problem prob)
 (:domain barman)
 (:objects 
      shaker129 - shaker
      left right - hand
      shot69 shot68 - shot
      ingredient204 ingredient318 ingredient120 - ingredient
      cocktail1 - cocktail
      dispenser312 dispenser45 dispenser64 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker129)
  (ontable shot69)
  (ontable shot68)
  (dispenses dispenser312 ingredient204)
  (dispenses dispenser45 ingredient318)
  (dispenses dispenser64 ingredient120)
  (clean shaker129)
  (clean shot69)
  (clean shot68)
  (empty shaker129)
  (empty shot69)
  (empty shot68)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker129 l0)
  (shaker-level shaker129 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient318)
  (cocktail-part2 cocktail1 ingredient120)
)
 (:goal
  (and
      (contains shot69 cocktail1)
)))
