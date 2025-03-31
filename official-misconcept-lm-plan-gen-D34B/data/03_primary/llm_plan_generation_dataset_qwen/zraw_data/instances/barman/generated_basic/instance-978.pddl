(define (problem prob)
 (:domain barman)
 (:objects 
      shaker151 - shaker
      left right - hand
      shot250 shot388 - shot
      ingredient463 ingredient2 ingredient270 ingredient5 - ingredient
      cocktail1 - cocktail
      dispenser449 dispenser216 dispenser91 dispenser311 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker151)
  (ontable shot250)
  (ontable shot388)
  (dispenses dispenser449 ingredient463)
  (dispenses dispenser216 ingredient2)
  (dispenses dispenser91 ingredient270)
  (dispenses dispenser311 ingredient5)
  (clean shaker151)
  (clean shot250)
  (clean shot388)
  (empty shaker151)
  (empty shot250)
  (empty shot388)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker151 l0)
  (shaker-level shaker151 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient463)
  (cocktail-part2 cocktail1 ingredient270)
)
 (:goal
  (and
      (contains shot250 cocktail1)
)))
