(define (problem prob)
 (:domain barman)
 (:objects 
      shaker16 - shaker
      left right - hand
      shot173 - shot
      ingredient434 ingredient361 - ingredient
      cocktail295 - cocktail
      dispenser17 dispenser144 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker16)
  (ontable shot173)
  (dispenses dispenser17 ingredient434)
  (dispenses dispenser144 ingredient361)
  (clean shaker16)
  (clean shot173)
  (empty shaker16)
  (empty shot173)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker16 l0)
  (shaker-level shaker16 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail295 ingredient434)
  (cocktail-part2 cocktail295 ingredient361)
)
 (:goal
  (and
      (contains shot173 cocktail295)
)))
