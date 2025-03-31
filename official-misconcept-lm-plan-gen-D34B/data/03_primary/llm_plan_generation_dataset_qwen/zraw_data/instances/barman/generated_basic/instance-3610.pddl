(define (problem prob)
 (:domain barman)
 (:objects 
      shaker204 - shaker
      left right - hand
      shot235 shot499 shot298 - shot
      ingredient207 ingredient254 ingredient350 - ingredient
      cocktail289 - cocktail
      dispenser27 dispenser91 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker204)
  (ontable shot235)
  (ontable shot499)
  (ontable shot298)
  (dispenses dispenser27 ingredient207)
  (dispenses dispenser91 ingredient254)
  (dispenses dispenser478 ingredient350)
  (clean shaker204)
  (clean shot235)
  (clean shot499)
  (clean shot298)
  (empty shaker204)
  (empty shot235)
  (empty shot499)
  (empty shot298)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker204 l0)
  (shaker-level shaker204 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail289 ingredient254)
  (cocktail-part2 cocktail289 ingredient350)
)
 (:goal
  (and
      (contains shot235 cocktail289)
      (contains shot499 cocktail289)
)))
