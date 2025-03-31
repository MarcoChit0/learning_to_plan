(define (problem prob)
 (:domain barman)
 (:objects 
      shaker458 - shaker
      left right - hand
      shot221 shot147 - shot
      ingredient158 ingredient294 ingredient280 - ingredient
      cocktail143 - cocktail
      dispenser150 dispenser114 dispenser5 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker458)
  (ontable shot221)
  (ontable shot147)
  (dispenses dispenser150 ingredient158)
  (dispenses dispenser114 ingredient294)
  (dispenses dispenser5 ingredient280)
  (clean shaker458)
  (clean shot221)
  (clean shot147)
  (empty shaker458)
  (empty shot221)
  (empty shot147)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker458 l0)
  (shaker-level shaker458 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail143 ingredient158)
  (cocktail-part2 cocktail143 ingredient280)
)
 (:goal
  (and
      (contains shot221 cocktail143)
)))
