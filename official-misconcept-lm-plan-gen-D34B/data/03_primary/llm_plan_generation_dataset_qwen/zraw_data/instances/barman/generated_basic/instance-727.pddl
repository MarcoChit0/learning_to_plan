(define (problem prob)
 (:domain barman)
 (:objects 
      shaker318 - shaker
      left right - hand
      shot347 - shot
      ingredient367 ingredient35 ingredient483 - ingredient
      cocktail1 - cocktail
      dispenser258 dispenser15 dispenser314 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker318)
  (ontable shot347)
  (dispenses dispenser258 ingredient367)
  (dispenses dispenser15 ingredient35)
  (dispenses dispenser314 ingredient483)
  (clean shaker318)
  (clean shot347)
  (empty shaker318)
  (empty shot347)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker318 l0)
  (shaker-level shaker318 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient35)
  (cocktail-part2 cocktail1 ingredient483)
)
 (:goal
  (and
      (contains shot347 cocktail1)
)))
