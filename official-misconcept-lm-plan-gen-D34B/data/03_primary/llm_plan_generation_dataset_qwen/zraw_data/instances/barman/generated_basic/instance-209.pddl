(define (problem prob)
 (:domain barman)
 (:objects 
      shaker165 - shaker
      left right - hand
      shot154 - shot
      ingredient7 ingredient456 ingredient224 - ingredient
      cocktail1 - cocktail
      dispenser227 dispenser36 dispenser453 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker165)
  (ontable shot154)
  (dispenses dispenser227 ingredient7)
  (dispenses dispenser36 ingredient456)
  (dispenses dispenser453 ingredient224)
  (clean shaker165)
  (clean shot154)
  (empty shaker165)
  (empty shot154)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker165 l0)
  (shaker-level shaker165 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient7)
  (cocktail-part2 cocktail1 ingredient456)
)
 (:goal
  (and
      (contains shot154 cocktail1)
)))
