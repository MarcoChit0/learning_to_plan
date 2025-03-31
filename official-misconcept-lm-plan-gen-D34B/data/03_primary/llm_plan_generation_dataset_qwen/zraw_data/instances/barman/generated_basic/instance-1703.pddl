(define (problem prob)
 (:domain barman)
 (:objects 
      shaker68 - shaker
      left right - hand
      shot433 - shot
      ingredient237 ingredient433 ingredient190 ingredient23 - ingredient
      cocktail346 - cocktail
      dispenser417 dispenser176 dispenser224 dispenser447 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker68)
  (ontable shot433)
  (dispenses dispenser417 ingredient237)
  (dispenses dispenser176 ingredient433)
  (dispenses dispenser224 ingredient190)
  (dispenses dispenser447 ingredient23)
  (clean shaker68)
  (clean shot433)
  (empty shaker68)
  (empty shot433)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker68 l0)
  (shaker-level shaker68 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail346 ingredient433)
  (cocktail-part2 cocktail346 ingredient190)
)
 (:goal
  (and
      (contains shot433 cocktail346)
)))
