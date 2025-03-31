(define (problem prob)
 (:domain barman)
 (:objects 
      shaker183 - shaker
      left right - hand
      shot242 - shot
      ingredient142 ingredient276 ingredient283 ingredient236 - ingredient
      cocktail406 - cocktail
      dispenser337 dispenser326 dispenser176 dispenser492 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker183)
  (ontable shot242)
  (dispenses dispenser337 ingredient142)
  (dispenses dispenser326 ingredient276)
  (dispenses dispenser176 ingredient283)
  (dispenses dispenser492 ingredient236)
  (clean shaker183)
  (clean shot242)
  (empty shaker183)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker183 l0)
  (shaker-level shaker183 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail406 ingredient142)
  (cocktail-part2 cocktail406 ingredient276)
)
 (:goal
  (and
      (contains shot242 cocktail406)
)))
