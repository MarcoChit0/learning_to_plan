(define (problem prob)
 (:domain barman)
 (:objects 
      shaker437 - shaker
      left right - hand
      shot155 - shot
      ingredient162 ingredient319 ingredient365 ingredient92 - ingredient
      cocktail293 - cocktail
      dispenser1 dispenser84 dispenser272 dispenser338 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker437)
  (ontable shot155)
  (dispenses dispenser1 ingredient162)
  (dispenses dispenser84 ingredient319)
  (dispenses dispenser272 ingredient365)
  (dispenses dispenser338 ingredient92)
  (clean shaker437)
  (clean shot155)
  (empty shaker437)
  (empty shot155)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker437 l0)
  (shaker-level shaker437 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail293 ingredient162)
  (cocktail-part2 cocktail293 ingredient92)
)
 (:goal
  (and
      (contains shot155 cocktail293)
)))
