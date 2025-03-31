(define (problem prob)
 (:domain barman)
 (:objects 
      shaker440 - shaker
      left right - hand
      shot66 - shot
      ingredient336 ingredient409 - ingredient
      cocktail274 - cocktail
      dispenser304 dispenser34 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker440)
  (ontable shot66)
  (dispenses dispenser304 ingredient336)
  (dispenses dispenser34 ingredient409)
  (clean shaker440)
  (clean shot66)
  (empty shaker440)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker440 l0)
  (shaker-level shaker440 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail274 ingredient336)
  (cocktail-part2 cocktail274 ingredient409)
)
 (:goal
  (and
      (contains shot66 cocktail274)
)))
