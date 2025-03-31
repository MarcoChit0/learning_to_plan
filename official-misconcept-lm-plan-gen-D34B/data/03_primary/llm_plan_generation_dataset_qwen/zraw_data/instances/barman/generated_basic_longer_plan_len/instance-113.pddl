(define (problem prob)
 (:domain barman)
 (:objects 
      shaker38 - shaker
      left right - hand
      shot84 shot257 shot278 - shot
      ingredient295 ingredient306 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser393 dispenser370 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker38)
  (ontable shot84)
  (ontable shot257)
  (ontable shot278)
  (dispenses dispenser393 ingredient295)
  (dispenses dispenser370 ingredient306)
  (clean shaker38)
  (clean shot84)
  (clean shot257)
  (clean shot278)
  (empty shaker38)
  (empty shot84)
  (empty shot257)
  (empty shot278)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker38 l0)
  (shaker-level shaker38 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient295)
  (cocktail-part2 cocktail1 ingredient306)
  (cocktail-part1 cocktail2 ingredient295)
  (cocktail-part2 cocktail2 ingredient306)
)
 (:goal
  (and
      (contains shot84 cocktail2)
      (contains shot257 cocktail1)
)))
