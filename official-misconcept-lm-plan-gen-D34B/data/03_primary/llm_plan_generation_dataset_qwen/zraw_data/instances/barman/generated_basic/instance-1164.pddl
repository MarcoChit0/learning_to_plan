(define (problem prob)
 (:domain barman)
 (:objects 
      shaker268 - shaker
      left right - hand
      shot86 shot370 - shot
      ingredient359 ingredient482 ingredient439 ingredient370 - ingredient
      cocktail1 - cocktail
      dispenser241 dispenser113 dispenser83 dispenser2 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker268)
  (ontable shot86)
  (ontable shot370)
  (dispenses dispenser241 ingredient359)
  (dispenses dispenser113 ingredient482)
  (dispenses dispenser83 ingredient439)
  (dispenses dispenser2 ingredient370)
  (clean shaker268)
  (clean shot86)
  (clean shot370)
  (empty shaker268)
  (empty shot86)
  (empty shot370)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker268 l0)
  (shaker-level shaker268 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient370)
  (cocktail-part2 cocktail1 ingredient482)
)
 (:goal
  (and
      (contains shot86 cocktail1)
)))
