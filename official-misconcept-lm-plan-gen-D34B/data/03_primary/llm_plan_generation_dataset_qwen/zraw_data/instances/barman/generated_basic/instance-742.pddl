(define (problem prob)
 (:domain barman)
 (:objects 
      shaker215 - shaker
      left right - hand
      shot159 shot41 - shot
      ingredient199 ingredient374 ingredient299 - ingredient
      cocktail1 - cocktail
      dispenser417 dispenser485 dispenser371 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker215)
  (ontable shot159)
  (ontable shot41)
  (dispenses dispenser417 ingredient199)
  (dispenses dispenser485 ingredient374)
  (dispenses dispenser371 ingredient299)
  (clean shaker215)
  (clean shot159)
  (clean shot41)
  (empty shaker215)
  (empty shot159)
  (empty shot41)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker215 l0)
  (shaker-level shaker215 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient199)
  (cocktail-part2 cocktail1 ingredient299)
)
 (:goal
  (and
      (contains shot159 cocktail1)
)))
