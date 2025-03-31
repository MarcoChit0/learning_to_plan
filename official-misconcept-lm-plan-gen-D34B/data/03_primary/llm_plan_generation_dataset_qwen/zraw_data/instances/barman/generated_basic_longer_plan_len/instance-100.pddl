(define (problem prob)
 (:domain barman)
 (:objects 
      shaker246 - shaker
      left right - hand
      shot223 shot312 shot390 - shot
      ingredient289 ingredient30 ingredient147 ingredient247 - ingredient
      cocktail1 - cocktail
      dispenser456 dispenser314 dispenser479 dispenser311 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker246)
  (ontable shot223)
  (ontable shot312)
  (ontable shot390)
  (dispenses dispenser456 ingredient289)
  (dispenses dispenser314 ingredient30)
  (dispenses dispenser479 ingredient147)
  (dispenses dispenser311 ingredient247)
  (clean shaker246)
  (clean shot223)
  (clean shot312)
  (clean shot390)
  (empty shaker246)
  (empty shot223)
  (empty shot312)
  (empty shot390)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker246 l0)
  (shaker-level shaker246 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient30)
  (cocktail-part2 cocktail1 ingredient289)
)
 (:goal
  (and
      (contains shot223 cocktail1)
      (contains shot312 ingredient147)
)))
