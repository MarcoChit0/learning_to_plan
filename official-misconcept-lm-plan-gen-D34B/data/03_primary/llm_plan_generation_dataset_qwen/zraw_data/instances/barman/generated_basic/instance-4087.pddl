(define (problem prob)
 (:domain barman)
 (:objects 
      shaker129 - shaker
      left right - hand
      shot15 shot391 - shot
      ingredient72 ingredient485 ingredient242 ingredient286 - ingredient
      cocktail12 - cocktail
      dispenser162 dispenser441 dispenser60 dispenser220 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker129)
  (ontable shot15)
  (ontable shot391)
  (dispenses dispenser162 ingredient72)
  (dispenses dispenser441 ingredient485)
  (dispenses dispenser60 ingredient242)
  (dispenses dispenser220 ingredient286)
  (clean shaker129)
  (clean shot15)
  (clean shot391)
  (empty shaker129)
  (empty shot15)
  (empty shot391)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker129 l0)
  (shaker-level shaker129 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail12 ingredient286)
  (cocktail-part2 cocktail12 ingredient485)
)
 (:goal
  (and
      (contains shot15 cocktail12)
)))
