(define (problem prob)
 (:domain barman)
 (:objects 
      shaker350 - shaker
      left right - hand
      shot237 shot285 - shot
      ingredient403 ingredient361 ingredient380 - ingredient
      cocktail90 - cocktail
      dispenser1 dispenser474 dispenser450 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker350)
  (ontable shot237)
  (ontable shot285)
  (dispenses dispenser1 ingredient403)
  (dispenses dispenser474 ingredient361)
  (dispenses dispenser450 ingredient380)
  (clean shaker350)
  (clean shot237)
  (clean shot285)
  (empty shaker350)
  (empty shot237)
  (empty shot285)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker350 l0)
  (shaker-level shaker350 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail90 ingredient403)
  (cocktail-part2 cocktail90 ingredient361)
)
 (:goal
  (and
      (contains shot237 cocktail90)
)))
