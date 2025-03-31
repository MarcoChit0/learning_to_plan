(define (problem prob)
 (:domain barman)
 (:objects 
      shaker383 - shaker
      left right - hand
      shot160 shot220 - shot
      ingredient238 ingredient176 - ingredient
      cocktail325 - cocktail
      dispenser186 dispenser262 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker383)
  (ontable shot160)
  (ontable shot220)
  (dispenses dispenser186 ingredient238)
  (dispenses dispenser262 ingredient176)
  (clean shaker383)
  (clean shot160)
  (clean shot220)
  (empty shaker383)
  (empty shot160)
  (empty shot220)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker383 l0)
  (shaker-level shaker383 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail325 ingredient176)
  (cocktail-part2 cocktail325 ingredient238)
)
 (:goal
  (and
      (contains shot160 cocktail325)
)))
