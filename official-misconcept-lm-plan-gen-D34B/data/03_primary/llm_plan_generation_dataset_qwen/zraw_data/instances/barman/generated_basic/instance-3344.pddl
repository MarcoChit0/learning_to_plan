(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot349 shot77 - shot
      ingredient200 ingredient57 - ingredient
      cocktail196 - cocktail
      dispenser15 dispenser6 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot349)
  (ontable shot77)
  (dispenses dispenser15 ingredient200)
  (dispenses dispenser6 ingredient57)
  (clean shaker221)
  (clean shot349)
  (clean shot77)
  (empty shaker221)
  (empty shot349)
  (empty shot77)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail196 ingredient200)
  (cocktail-part2 cocktail196 ingredient57)
)
 (:goal
  (and
      (contains shot349 cocktail196)
)))
