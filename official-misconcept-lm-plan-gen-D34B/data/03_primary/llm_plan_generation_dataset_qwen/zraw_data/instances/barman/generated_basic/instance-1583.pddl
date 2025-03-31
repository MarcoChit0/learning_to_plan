(define (problem prob)
 (:domain barman)
 (:objects 
      shaker251 - shaker
      left right - hand
      shot176 shot498 - shot
      ingredient256 ingredient303 - ingredient
      cocktail86 - cocktail
      dispenser469 dispenser251 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker251)
  (ontable shot176)
  (ontable shot498)
  (dispenses dispenser469 ingredient256)
  (dispenses dispenser251 ingredient303)
  (clean shaker251)
  (clean shot176)
  (clean shot498)
  (empty shaker251)
  (empty shot176)
  (empty shot498)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker251 l0)
  (shaker-level shaker251 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail86 ingredient303)
  (cocktail-part2 cocktail86 ingredient256)
)
 (:goal
  (and
      (contains shot176 cocktail86)
)))
