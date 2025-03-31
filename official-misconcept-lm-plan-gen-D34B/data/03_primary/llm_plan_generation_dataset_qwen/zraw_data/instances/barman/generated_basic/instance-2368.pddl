(define (problem prob)
 (:domain barman)
 (:objects 
      shaker266 - shaker
      left right - hand
      shot116 - shot
      ingredient54 ingredient165 ingredient142 ingredient58 - ingredient
      cocktail160 - cocktail
      dispenser249 dispenser418 dispenser177 dispenser113 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker266)
  (ontable shot116)
  (dispenses dispenser249 ingredient54)
  (dispenses dispenser418 ingredient165)
  (dispenses dispenser177 ingredient142)
  (dispenses dispenser113 ingredient58)
  (clean shaker266)
  (clean shot116)
  (empty shaker266)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker266 l0)
  (shaker-level shaker266 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail160 ingredient58)
  (cocktail-part2 cocktail160 ingredient165)
)
 (:goal
  (and
      (contains shot116 cocktail160)
)))
