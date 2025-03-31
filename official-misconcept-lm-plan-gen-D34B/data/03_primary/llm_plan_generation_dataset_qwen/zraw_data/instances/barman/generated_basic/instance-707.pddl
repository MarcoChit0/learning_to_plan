(define (problem prob)
 (:domain barman)
 (:objects 
      shaker349 - shaker
      left right - hand
      shot38 shot106 shot200 - shot
      ingredient114 ingredient308 ingredient134 ingredient85 - ingredient
      cocktail1 - cocktail
      dispenser444 dispenser190 dispenser155 dispenser186 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker349)
  (ontable shot38)
  (ontable shot106)
  (ontable shot200)
  (dispenses dispenser444 ingredient114)
  (dispenses dispenser190 ingredient308)
  (dispenses dispenser155 ingredient134)
  (dispenses dispenser186 ingredient85)
  (clean shaker349)
  (clean shot38)
  (clean shot106)
  (clean shot200)
  (empty shaker349)
  (empty shot38)
  (empty shot106)
  (empty shot200)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker349 l0)
  (shaker-level shaker349 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient308)
  (cocktail-part2 cocktail1 ingredient114)
)
 (:goal
  (and
      (contains shot38 cocktail1)
      (contains shot106 cocktail1)
)))
