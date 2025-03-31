(define (problem prob)
 (:domain barman)
 (:objects 
      shaker187 - shaker
      left right - hand
      shot144 shot93 - shot
      ingredient18 ingredient221 ingredient298 - ingredient
      cocktail1 - cocktail
      dispenser489 dispenser295 dispenser477 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker187)
  (ontable shot144)
  (ontable shot93)
  (dispenses dispenser489 ingredient18)
  (dispenses dispenser295 ingredient221)
  (dispenses dispenser477 ingredient298)
  (clean shaker187)
  (clean shot144)
  (clean shot93)
  (empty shaker187)
  (empty shot144)
  (empty shot93)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker187 l0)
  (shaker-level shaker187 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient298)
  (cocktail-part2 cocktail1 ingredient221)
)
 (:goal
  (and
      (contains shot144 cocktail1)
)))
