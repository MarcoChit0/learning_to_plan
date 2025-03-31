(define (problem prob)
 (:domain barman)
 (:objects 
      shaker339 - shaker
      left right - hand
      shot244 shot240 - shot
      ingredient478 ingredient496 ingredient433 ingredient216 - ingredient
      cocktail1 - cocktail
      dispenser250 dispenser248 dispenser91 dispenser350 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker339)
  (ontable shot244)
  (ontable shot240)
  (dispenses dispenser250 ingredient478)
  (dispenses dispenser248 ingredient496)
  (dispenses dispenser91 ingredient433)
  (dispenses dispenser350 ingredient216)
  (clean shaker339)
  (clean shot244)
  (clean shot240)
  (empty shaker339)
  (empty shot244)
  (empty shot240)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker339 l0)
  (shaker-level shaker339 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient478)
  (cocktail-part2 cocktail1 ingredient496)
)
 (:goal
  (and
      (contains shot244 cocktail1)
)))
