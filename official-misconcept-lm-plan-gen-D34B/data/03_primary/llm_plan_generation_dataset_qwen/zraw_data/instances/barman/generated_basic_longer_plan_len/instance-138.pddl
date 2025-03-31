(define (problem prob)
 (:domain barman)
 (:objects 
      shaker238 - shaker
      left right - hand
      shot413 shot96 - shot
      ingredient440 ingredient253 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser174 dispenser285 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker238)
  (ontable shot413)
  (ontable shot96)
  (dispenses dispenser174 ingredient440)
  (dispenses dispenser285 ingredient253)
  (clean shaker238)
  (clean shot413)
  (clean shot96)
  (empty shaker238)
  (empty shot413)
  (empty shot96)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker238 l0)
  (shaker-level shaker238 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient253)
  (cocktail-part2 cocktail1 ingredient440)
  (cocktail-part1 cocktail2 ingredient253)
  (cocktail-part2 cocktail2 ingredient440)
)
 (:goal
  (and
      (contains shot413 cocktail1)
      (contains shot96 cocktail2)
)))
