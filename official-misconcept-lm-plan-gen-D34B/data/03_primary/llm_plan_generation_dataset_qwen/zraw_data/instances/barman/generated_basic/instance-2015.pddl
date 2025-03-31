(define (problem prob)
 (:domain barman)
 (:objects 
      shaker260 - shaker
      left right - hand
      shot54 shot36 shot137 - shot
      ingredient323 ingredient144 ingredient184 - ingredient
      cocktail278 - cocktail
      dispenser158 dispenser104 dispenser205 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker260)
  (ontable shot54)
  (ontable shot36)
  (ontable shot137)
  (dispenses dispenser158 ingredient323)
  (dispenses dispenser104 ingredient144)
  (dispenses dispenser205 ingredient184)
  (clean shaker260)
  (clean shot54)
  (clean shot36)
  (clean shot137)
  (empty shaker260)
  (empty shot54)
  (empty shot36)
  (empty shot137)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker260 l0)
  (shaker-level shaker260 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail278 ingredient323)
  (cocktail-part2 cocktail278 ingredient184)
)
 (:goal
  (and
      (contains shot54 cocktail278)
      (contains shot36 cocktail278)
)))
