(define (problem prob)
 (:domain barman)
 (:objects 
      shaker86 - shaker
      left right - hand
      shot471 shot213 shot138 - shot
      ingredient163 ingredient165 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser449 dispenser429 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker86)
  (ontable shot471)
  (ontable shot213)
  (ontable shot138)
  (dispenses dispenser449 ingredient163)
  (dispenses dispenser429 ingredient165)
  (clean shaker86)
  (clean shot471)
  (clean shot213)
  (clean shot138)
  (empty shaker86)
  (empty shot471)
  (empty shot213)
  (empty shot138)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker86 l0)
  (shaker-level shaker86 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient163)
  (cocktail-part2 cocktail1 ingredient165)
  (cocktail-part1 cocktail2 ingredient163)
  (cocktail-part2 cocktail2 ingredient165)
)
 (:goal
  (and
      (contains shot471 cocktail2)
      (contains shot213 cocktail1)
)))
