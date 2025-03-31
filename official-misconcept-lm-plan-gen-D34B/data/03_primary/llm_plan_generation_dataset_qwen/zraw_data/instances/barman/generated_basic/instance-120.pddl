(define (problem prob)
 (:domain barman)
 (:objects 
      shaker301 - shaker
      left right - hand
      shot310 shot144 - shot
      ingredient74 ingredient217 - ingredient
      cocktail1 - cocktail
      dispenser49 dispenser359 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker301)
  (ontable shot310)
  (ontable shot144)
  (dispenses dispenser49 ingredient74)
  (dispenses dispenser359 ingredient217)
  (clean shaker301)
  (clean shot310)
  (clean shot144)
  (empty shaker301)
  (empty shot310)
  (empty shot144)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker301 l0)
  (shaker-level shaker301 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient217)
  (cocktail-part2 cocktail1 ingredient74)
)
 (:goal
  (and
      (contains shot310 cocktail1)
)))
