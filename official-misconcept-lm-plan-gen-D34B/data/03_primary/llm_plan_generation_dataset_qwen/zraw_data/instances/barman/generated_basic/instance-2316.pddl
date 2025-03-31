(define (problem prob)
 (:domain barman)
 (:objects 
      shaker244 - shaker
      left right - hand
      shot97 - shot
      ingredient175 ingredient200 - ingredient
      cocktail141 - cocktail
      dispenser403 dispenser83 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker244)
  (ontable shot97)
  (dispenses dispenser403 ingredient175)
  (dispenses dispenser83 ingredient200)
  (clean shaker244)
  (clean shot97)
  (empty shaker244)
  (empty shot97)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker244 l0)
  (shaker-level shaker244 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail141 ingredient200)
  (cocktail-part2 cocktail141 ingredient175)
)
 (:goal
  (and
      (contains shot97 cocktail141)
)))
