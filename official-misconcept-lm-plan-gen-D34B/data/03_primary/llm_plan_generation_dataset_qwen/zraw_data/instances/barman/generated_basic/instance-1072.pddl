(define (problem prob)
 (:domain barman)
 (:objects 
      shaker49 - shaker
      left right - hand
      shot72 shot364 shot13 - shot
      ingredient97 ingredient144 ingredient349 ingredient454 - ingredient
      cocktail1 - cocktail
      dispenser418 dispenser406 dispenser38 dispenser101 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker49)
  (ontable shot72)
  (ontable shot364)
  (ontable shot13)
  (dispenses dispenser418 ingredient97)
  (dispenses dispenser406 ingredient144)
  (dispenses dispenser38 ingredient349)
  (dispenses dispenser101 ingredient454)
  (clean shaker49)
  (clean shot72)
  (clean shot364)
  (clean shot13)
  (empty shaker49)
  (empty shot72)
  (empty shot364)
  (empty shot13)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker49 l0)
  (shaker-level shaker49 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient97)
  (cocktail-part2 cocktail1 ingredient454)
)
 (:goal
  (and
      (contains shot72 cocktail1)
      (contains shot364 cocktail1)
)))
