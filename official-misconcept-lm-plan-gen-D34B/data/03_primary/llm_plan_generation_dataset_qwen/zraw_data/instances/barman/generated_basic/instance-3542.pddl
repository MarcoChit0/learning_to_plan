(define (problem prob)
 (:domain barman)
 (:objects 
      shaker364 - shaker
      left right - hand
      shot421 - shot
      ingredient291 ingredient62 ingredient336 - ingredient
      cocktail120 - cocktail
      dispenser289 dispenser104 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker364)
  (ontable shot421)
  (dispenses dispenser289 ingredient291)
  (dispenses dispenser104 ingredient62)
  (dispenses dispenser385 ingredient336)
  (clean shaker364)
  (clean shot421)
  (empty shaker364)
  (empty shot421)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker364 l0)
  (shaker-level shaker364 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail120 ingredient62)
  (cocktail-part2 cocktail120 ingredient291)
)
 (:goal
  (and
      (contains shot421 cocktail120)
)))
