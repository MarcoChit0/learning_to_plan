(define (problem prob)
 (:domain barman)
 (:objects 
      shaker61 - shaker
      left right - hand
      shot18 - shot
      ingredient363 ingredient217 ingredient26 ingredient189 - ingredient
      cocktail50 - cocktail
      dispenser116 dispenser207 dispenser249 dispenser206 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker61)
  (ontable shot18)
  (dispenses dispenser116 ingredient363)
  (dispenses dispenser207 ingredient217)
  (dispenses dispenser249 ingredient26)
  (dispenses dispenser206 ingredient189)
  (clean shaker61)
  (clean shot18)
  (empty shaker61)
  (empty shot18)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker61 l0)
  (shaker-level shaker61 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail50 ingredient189)
  (cocktail-part2 cocktail50 ingredient363)
)
 (:goal
  (and
      (contains shot18 cocktail50)
)))
