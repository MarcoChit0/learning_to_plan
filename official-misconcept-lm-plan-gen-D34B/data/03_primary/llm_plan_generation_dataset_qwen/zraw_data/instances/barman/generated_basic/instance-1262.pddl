(define (problem prob)
 (:domain barman)
 (:objects 
      shaker439 - shaker
      left right - hand
      shot340 shot95 shot489 - shot
      ingredient82 ingredient268 ingredient182 ingredient208 - ingredient
      cocktail133 - cocktail
      dispenser316 dispenser41 dispenser443 dispenser229 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker439)
  (ontable shot340)
  (ontable shot95)
  (ontable shot489)
  (dispenses dispenser316 ingredient82)
  (dispenses dispenser41 ingredient268)
  (dispenses dispenser443 ingredient182)
  (dispenses dispenser229 ingredient208)
  (clean shaker439)
  (clean shot340)
  (clean shot95)
  (clean shot489)
  (empty shaker439)
  (empty shot340)
  (empty shot95)
  (empty shot489)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker439 l0)
  (shaker-level shaker439 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail133 ingredient182)
  (cocktail-part2 cocktail133 ingredient82)
)
 (:goal
  (and
      (contains shot340 cocktail133)
      (contains shot95 cocktail133)
)))
