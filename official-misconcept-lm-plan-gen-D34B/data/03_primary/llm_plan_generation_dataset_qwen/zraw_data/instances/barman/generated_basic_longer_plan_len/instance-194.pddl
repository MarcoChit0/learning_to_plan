(define (problem prob)
 (:domain barman)
 (:objects 
      shaker365 - shaker
      left right - hand
      shot96 shot354 shot30 - shot
      ingredient471 ingredient86 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser22 dispenser88 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker365)
  (ontable shot96)
  (ontable shot354)
  (ontable shot30)
  (dispenses dispenser22 ingredient471)
  (dispenses dispenser88 ingredient86)
  (clean shaker365)
  (clean shot96)
  (clean shot354)
  (clean shot30)
  (empty shaker365)
  (empty shot96)
  (empty shot354)
  (empty shot30)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker365 l0)
  (shaker-level shaker365 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient86)
  (cocktail-part2 cocktail1 ingredient471)
  (cocktail-part1 cocktail2 ingredient471)
  (cocktail-part2 cocktail2 ingredient86)
)
 (:goal
  (and
      (contains shot96 cocktail1)
      (contains shot354 cocktail2)
)))
