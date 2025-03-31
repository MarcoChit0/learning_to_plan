(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot0 - shot
      ingredient336 ingredient332 - ingredient
      cocktail176 - cocktail
      dispenser321 dispenser80 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot0)
  (dispenses dispenser321 ingredient336)
  (dispenses dispenser80 ingredient332)
  (clean shaker426)
  (clean shot0)
  (empty shaker426)
  (empty shot0)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail176 ingredient336)
  (cocktail-part2 cocktail176 ingredient332)
)
 (:goal
  (and
      (contains shot0 cocktail176)
)))
