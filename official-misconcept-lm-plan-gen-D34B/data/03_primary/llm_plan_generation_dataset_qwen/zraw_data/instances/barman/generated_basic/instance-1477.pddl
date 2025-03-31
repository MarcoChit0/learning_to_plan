(define (problem prob)
 (:domain barman)
 (:objects 
      shaker38 - shaker
      left right - hand
      shot144 shot287 - shot
      ingredient41 ingredient346 ingredient15 ingredient302 - ingredient
      cocktail274 - cocktail
      dispenser483 dispenser13 dispenser379 dispenser267 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker38)
  (ontable shot144)
  (ontable shot287)
  (dispenses dispenser483 ingredient41)
  (dispenses dispenser13 ingredient346)
  (dispenses dispenser379 ingredient15)
  (dispenses dispenser267 ingredient302)
  (clean shaker38)
  (clean shot144)
  (clean shot287)
  (empty shaker38)
  (empty shot144)
  (empty shot287)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker38 l0)
  (shaker-level shaker38 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail274 ingredient302)
  (cocktail-part2 cocktail274 ingredient346)
)
 (:goal
  (and
      (contains shot144 cocktail274)
)))
