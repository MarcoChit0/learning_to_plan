(define (problem prob)
 (:domain barman)
 (:objects 
      shaker365 - shaker
      left right - hand
      shot305 - shot
      ingredient371 ingredient339 ingredient334 - ingredient
      cocktail1 - cocktail
      dispenser251 dispenser228 dispenser256 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker365)
  (ontable shot305)
  (dispenses dispenser251 ingredient371)
  (dispenses dispenser228 ingredient339)
  (dispenses dispenser256 ingredient334)
  (clean shaker365)
  (clean shot305)
  (empty shaker365)
  (empty shot305)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker365 l0)
  (shaker-level shaker365 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient371)
  (cocktail-part2 cocktail1 ingredient339)
)
 (:goal
  (and
      (contains shot305 cocktail1)
)))
