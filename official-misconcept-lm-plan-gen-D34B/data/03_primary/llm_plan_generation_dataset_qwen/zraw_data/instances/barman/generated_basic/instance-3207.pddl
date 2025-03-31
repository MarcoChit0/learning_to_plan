(define (problem prob)
 (:domain barman)
 (:objects 
      shaker38 - shaker
      left right - hand
      shot316 shot106 shot183 shot223 - shot
      ingredient262 ingredient220 ingredient263 - ingredient
      cocktail227 - cocktail
      dispenser401 dispenser492 dispenser51 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker38)
  (ontable shot316)
  (ontable shot106)
  (ontable shot183)
  (ontable shot223)
  (dispenses dispenser401 ingredient262)
  (dispenses dispenser492 ingredient220)
  (dispenses dispenser51 ingredient263)
  (clean shaker38)
  (clean shot316)
  (clean shot106)
  (clean shot183)
  (clean shot223)
  (empty shaker38)
  (empty shot316)
  (empty shot106)
  (empty shot183)
  (empty shot223)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker38 l0)
  (shaker-level shaker38 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail227 ingredient263)
  (cocktail-part2 cocktail227 ingredient262)
)
 (:goal
  (and
      (contains shot316 cocktail227)
      (contains shot106 ingredient263)
      (contains shot183 cocktail227)
)))
