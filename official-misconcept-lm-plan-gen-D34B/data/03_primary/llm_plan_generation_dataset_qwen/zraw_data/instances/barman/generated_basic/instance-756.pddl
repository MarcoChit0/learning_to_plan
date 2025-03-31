(define (problem prob)
 (:domain barman)
 (:objects 
      shaker488 - shaker
      left right - hand
      shot300 shot423 shot30 - shot
      ingredient348 ingredient451 ingredient49 - ingredient
      cocktail1 - cocktail
      dispenser394 dispenser170 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker488)
  (ontable shot300)
  (ontable shot423)
  (ontable shot30)
  (dispenses dispenser394 ingredient348)
  (dispenses dispenser170 ingredient451)
  (dispenses dispenser214 ingredient49)
  (clean shaker488)
  (clean shot300)
  (clean shot423)
  (clean shot30)
  (empty shaker488)
  (empty shot300)
  (empty shot423)
  (empty shot30)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker488 l0)
  (shaker-level shaker488 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient348)
  (cocktail-part2 cocktail1 ingredient49)
)
 (:goal
  (and
      (contains shot300 cocktail1)
      (contains shot423 cocktail1)
)))
