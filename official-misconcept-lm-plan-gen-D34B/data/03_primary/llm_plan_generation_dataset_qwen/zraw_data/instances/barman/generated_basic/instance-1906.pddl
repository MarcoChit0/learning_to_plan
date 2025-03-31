(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot32 - shot
      ingredient221 ingredient154 ingredient285 ingredient489 - ingredient
      cocktail197 - cocktail
      dispenser313 dispenser169 dispenser240 dispenser351 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot32)
  (dispenses dispenser313 ingredient221)
  (dispenses dispenser169 ingredient154)
  (dispenses dispenser240 ingredient285)
  (dispenses dispenser351 ingredient489)
  (clean shaker236)
  (clean shot32)
  (empty shaker236)
  (empty shot32)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail197 ingredient221)
  (cocktail-part2 cocktail197 ingredient154)
)
 (:goal
  (and
      (contains shot32 cocktail197)
)))
