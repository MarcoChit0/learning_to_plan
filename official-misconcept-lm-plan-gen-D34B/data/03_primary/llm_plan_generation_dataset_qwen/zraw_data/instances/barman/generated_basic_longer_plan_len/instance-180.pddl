(define (problem prob)
 (:domain barman)
 (:objects 
      shaker98 - shaker
      left right - hand
      shot225 shot357 shot150 - shot
      ingredient172 ingredient237 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser405 dispenser20 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker98)
  (ontable shot225)
  (ontable shot357)
  (ontable shot150)
  (dispenses dispenser405 ingredient172)
  (dispenses dispenser20 ingredient237)
  (clean shaker98)
  (clean shot225)
  (clean shot357)
  (clean shot150)
  (empty shaker98)
  (empty shot225)
  (empty shot357)
  (empty shot150)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker98 l0)
  (shaker-level shaker98 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient237)
  (cocktail-part2 cocktail1 ingredient172)
  (cocktail-part1 cocktail2 ingredient237)
  (cocktail-part2 cocktail2 ingredient172)
)
 (:goal
  (and
      (contains shot225 cocktail1)
      (contains shot357 cocktail2)
)))
