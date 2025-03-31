(define (problem prob)
 (:domain barman)
 (:objects 
      shaker26 - shaker
      left right - hand
      shot7 shot449 shot173 - shot
      ingredient388 ingredient303 ingredient469 ingredient326 - ingredient
      cocktail258 - cocktail
      dispenser167 dispenser156 dispenser497 dispenser498 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker26)
  (ontable shot7)
  (ontable shot449)
  (ontable shot173)
  (dispenses dispenser167 ingredient388)
  (dispenses dispenser156 ingredient303)
  (dispenses dispenser497 ingredient469)
  (dispenses dispenser498 ingredient326)
  (clean shaker26)
  (clean shot7)
  (clean shot449)
  (clean shot173)
  (empty shaker26)
  (empty shot7)
  (empty shot449)
  (empty shot173)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker26 l0)
  (shaker-level shaker26 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail258 ingredient388)
  (cocktail-part2 cocktail258 ingredient326)
)
 (:goal
  (and
      (contains shot7 cocktail258)
      (contains shot449 cocktail258)
)))
