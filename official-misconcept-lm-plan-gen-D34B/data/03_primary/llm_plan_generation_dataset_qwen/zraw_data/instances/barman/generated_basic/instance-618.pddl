(define (problem prob)
 (:domain barman)
 (:objects 
      shaker230 - shaker
      left right - hand
      shot430 shot126 - shot
      ingredient273 ingredient88 ingredient449 ingredient456 - ingredient
      cocktail1 - cocktail
      dispenser311 dispenser235 dispenser197 dispenser145 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker230)
  (ontable shot430)
  (ontable shot126)
  (dispenses dispenser311 ingredient273)
  (dispenses dispenser235 ingredient88)
  (dispenses dispenser197 ingredient449)
  (dispenses dispenser145 ingredient456)
  (clean shaker230)
  (clean shot430)
  (clean shot126)
  (empty shaker230)
  (empty shot430)
  (empty shot126)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker230 l0)
  (shaker-level shaker230 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient273)
  (cocktail-part2 cocktail1 ingredient449)
)
 (:goal
  (and
      (contains shot430 cocktail1)
)))
