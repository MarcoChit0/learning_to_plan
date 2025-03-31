(define (problem prob)
 (:domain barman)
 (:objects 
      shaker439 - shaker
      left right - hand
      shot279 - shot
      ingredient153 ingredient398 - ingredient
      cocktail1 - cocktail
      dispenser230 dispenser211 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker439)
  (ontable shot279)
  (dispenses dispenser230 ingredient153)
  (dispenses dispenser211 ingredient398)
  (clean shaker439)
  (clean shot279)
  (empty shaker439)
  (empty shot279)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker439 l0)
  (shaker-level shaker439 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient153)
  (cocktail-part2 cocktail1 ingredient398)
)
 (:goal
  (and
      (contains shot279 cocktail1)
)))
